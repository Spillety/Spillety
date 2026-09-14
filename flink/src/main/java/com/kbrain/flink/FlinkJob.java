package com.kbrain.flink;

import org.apache.flink.api.common.functions.MapFunction;
import org.apache.flink.api.common.serialization.DeserializationSchema;
import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.connector.kafka.source.KafkaSource;
import org.apache.flink.connector.kafka.source.enumerator.initializer.OffsetsInitializer;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.streaming.api.functions.sink.SinkFunction;
import com.kbrain.flink.config.RocksDBStateConfig;
import com.kbrain.flink.config.CheckpointConfig;
import com.kbrain.flink.config.EventTimeConfig;
import com.kbrain.flink.function.ExchangeEnrichment;
import com.kbrain.flink.function.GlobalDedupFilter;
import com.kbrain.flink.function.IncrementalPageRank;
import com.kbrain.flink.function.PowerLawHawkes;
import com.kbrain.flink.sink.MemgraphBoltSink;
import com.kbrain.flink.model.EdgeEvent;
import com.kbrain.flink.model.OnChainEvent;
import com.kbrain.flink.model.RankUpdate;
import com.kbrain.flink.model.HawkesUpdate;
import com.kbrain.flink.model.TemporalEdge;
import com.kbrain.flink.model.TemporalEdgeBuilder;

import java.nio.charset.StandardCharsets;
import java.util.Collections;

public class FlinkJob {

    private static final String DLQ_TOPIC = "dead-letter-events";

    // # ponytail: load from entity_resolution/config (pipeline.yaml exchange.*).
    private static final java.util.Set<String> EXCHANGE_SNAPSHOT = Collections.emptySet();

    private static String boltUri() {
        return System.getenv().getOrDefault("MEMGRAPH_BOLT_URI", "bolt://localhost:7687");
    }

    private static String boltUser() {
        String user = System.getenv("MEMGRAPH_USER");
        if (user == null) {
            throw new IllegalStateException("MEMGRAPH_USER is not set");
        }
        return user;
    }

    private static String boltPassword() {
        String password = System.getenv("MEMGRAPH_PASSWORD");
        if (password == null) {
            throw new IllegalStateException("MEMGRAPH_PASSWORD is not set");
        }
        return password;
    }

    private static String bootstrapServers() {
        return System.getenv().getOrDefault("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092");
    }

    public static void main(String[] args) throws Exception {
        StreamExecutionEnvironment env = RocksDBStateConfig.createEnvironment();

        CheckpointConfig.configure(env);

        KafkaSource<OnChainEvent> kafkaSource = KafkaSource.<OnChainEvent>builder()
            .setBootstrapServers(bootstrapServers())
            .setTopics("raw-events")
            .setGroupId("flink-stateful-group")
            .setStartingOffsets(OffsetsInitializer.earliest())
            .setValueOnlyDeserializer(new OnChainEventDeserializationSchema())
            .build();

        DataStream<OnChainEvent> ingested = env.fromSource(
            kafkaSource, EventTimeConfig.watermarkStrategy(), "kafka-raw-events");

        DataStream<OnChainEvent> deduped = ingested
            .keyBy(OnChainEvent::getEventId)
            .process(new GlobalDedupFilter());

        DataStream<OnChainEvent> enriched = deduped
            .map(new ExchangeEnrichment(EXCHANGE_SNAPSHOT));

        DataStream<HawkesUpdate> hawkesUpdates = enriched
            .keyBy(OnChainEvent::getFromAddress)
            .process(new PowerLawHawkes());

        DataStream<EdgeEvent> edges = enriched
            .map((MapFunction<OnChainEvent, EdgeEvent>) event ->
                new EdgeEvent(event.getFromAddress(), event.getToAddress(), event.getTimestamp()));

        DataStream<RankUpdate> rankUpdates = edges
            .keyBy(EdgeEvent::getSource)
            .process(new IncrementalPageRank());

        DataStream<TemporalEdge> temporalEdges = enriched
            .map(new TemporalEdgeBuilder());

        rankUpdates.addSink(MemgraphBoltSink.forRankUpdates(
            boltUri(), boltUser(), boltPassword(), bootstrapServers(), DLQ_TOPIC));
        hawkesUpdates.addSink(new ClickHouseUpdateSink());
        temporalEdges.addSink(MemgraphBoltSink.forTemporalEdges(
            boltUri(), boltUser(), boltPassword(), bootstrapServers(), DLQ_TOPIC));

        env.execute("Flink Stateful Stream Processing");
    }

    static class OnChainEventDeserializationSchema implements DeserializationSchema<OnChainEvent> {
        @Override
        public OnChainEvent deserialize(byte[] message) throws Exception {
            if (message.length > 0 && message[0] == '{') {
                return OnChainEvent.fromJson(new String(message, StandardCharsets.UTF_8));
            }
            return OnChainEvent.fromProtoBytes(message);
        }
        @Override
        public boolean isEndOfStream(OnChainEvent nextElement) { return false; }
        @Override
        public TypeInformation<OnChainEvent> getProducedType() {
            return TypeInformation.of(OnChainEvent.class);
        }
    }

    // # ponytail: full ClickHouse sink (JDBC batch) once tables exist.
    static class ClickHouseUpdateSink implements SinkFunction<HawkesUpdate> {
        @Override
        public void invoke(HawkesUpdate value, Context context) {
            throw new UnsupportedOperationException("ClickHouse sink not implemented");
        }
    }
}
