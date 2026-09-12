package com.kbrain.flink;

import org.apache.flink.connector.kafka.source.FlinkKafkaConsumer;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.api.common.serialization.DeserializationSchema;
import org.apache.flink.api.common.typeinfo.TypeInformation;
import com.kbrain.flink.config.RocksDBStateConfig;
import com.kbrain.flink.config.CheckpointConfig;
import com.kbrain.flink.function.IncrementalPageRank;
import com.kbrain.flink.function.PowerLawHawkes;
import com.kbrain.flink.model.EdgeEvent;
import com.kbrain.flink.model.OnChainEvent;
import com.kbrain.flink.model.RankUpdate;
import com.kbrain.flink.model.HawkesUpdate;

import java.util.Properties;

public class FlinkJob {

    public static void main(String[] args) throws Exception {
        StreamExecutionEnvironment env = RocksDBStateConfig.createEnvironment();

        CheckpointConfig.configure(env);

        Properties kafkaProps = new Properties();
        kafkaProps.setProperty("bootstrap.servers", "localhost:9092");
        kafkaProps.setProperty("group.id", "flink-stateful-group");

        FlinkKafkaConsumer<OnChainEvent> kafkaSource = new FlinkKafkaConsumer<>(
            "raw-events",
            new OnChainEventDeserializationSchema(),
            kafkaProps
        );
        kafkaSource.setStartFromEarliest();
        kafkaSource.setCommitOffsetsOnCheckpoints(true);

        DataStream<RankUpdate> rankUpdates = env
            .addSource(kafkaSource)
            .keyBy(event -> event.getFromAddress())
            .process(new IncrementalPageRank());

        DataStream<HawkesUpdate> hawkesUpdates = env
            .addSource(kafkaSource)
            .keyBy(event -> event.getFromAddress())
            .process(new PowerLawHawkes());

        rankUpdates.addSink(new org.apache.flink.streaming.api.functions.sink.SinkFunction<RankUpdate>() {
            @Override
            public void invoke(RankUpdate value, Context context) {}
        });

        hawkesUpdates.addSink(new org.apache.flink.streaming.api.functions.sink.SinkFunction<HawkesUpdate>() {
            @Override
            public void invoke(HawkesUpdate value, Context context) {}
        });

        env.execute("Flink Stateful Stream Processing");
    }

    static class OnChainEventDeserializationSchema implements DeserializationSchema<OnChainEvent> {
        @Override
        public OnChainEvent deserialize(byte[] message) throws Exception {
            return OnChainEvent.fromJson(new String(message));
        }
        @Override
        public boolean isEndOfStream(OnChainEvent nextElement) { return false; }
        @Override
        public TypeInformation<OnChainEvent> getProducedType() {
            return TypeInformation.of(OnChainEvent.class);
        }
    }
}
