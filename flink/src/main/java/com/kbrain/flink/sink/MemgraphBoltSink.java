package com.kbrain.flink.sink;

import org.apache.flink.runtime.state.FunctionInitializationContext;
import org.apache.flink.runtime.state.FunctionSnapshotContext;
import org.apache.flink.streaming.api.checkpoint.CheckpointedFunction;
import org.apache.flink.streaming.api.functions.sink.RichSinkFunction;
import org.apache.kafka.clients.producer.KafkaProducer;
import org.apache.kafka.clients.producer.ProducerRecord;
import org.neo4j.driver.AuthTokens;
import org.neo4j.driver.Driver;
import org.neo4j.driver.GraphDatabase;
import org.neo4j.driver.Session;
import org.neo4j.driver.Values;
import com.kbrain.flink.model.RankUpdate;
import com.kbrain.flink.model.TemporalEdge;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Semaphore;
import java.util.concurrent.TimeUnit;

public class MemgraphBoltSink<T> extends RichSinkFunction<T> implements CheckpointedFunction {
    public interface CypherMapper<E> extends Serializable {
        String cypher();
        Map<String, Object> params(E value);
        String recordId(E value);
    }
    private static final int MAX_RETRIES = 5;
    private static final long BASE_BACKOFF_MS = 200L;
    private static final int MAX_INFLIGHT = 1000;
    private final String boltUri;
    private final String username;
    private final String password;
    private final String dlqTopic;
    private final String bootstrapServers;
    private final CypherMapper<T> mapper;
    private transient Driver driver;
    private transient KafkaProducer<String, String> dlq;
    private transient ExecutorService writer;
    private transient Semaphore permits;
    private transient List<CompletableFuture<Void>> inflight;
    public MemgraphBoltSink(String boltUri, String username, String password, String bootstrapServers, String dlqTopic, CypherMapper<T> mapper) {
        this.boltUri = boltUri;
        this.username = username;
        this.password = password;
        this.bootstrapServers = bootstrapServers;
        this.dlqTopic = dlqTopic;
        this.mapper = mapper;
    }
    @Override
    public void open(org.apache.flink.configuration.Configuration parameters) {
        this.driver = GraphDatabase.driver(boltUri, AuthTokens.basic(username, password));
        Properties props = new Properties();
        props.put("bootstrap.servers", bootstrapServers);
        props.put("key.serializer", "org.apache.kafka.common.serialization.StringSerializer");
        props.put("value.serializer", "org.apache.kafka.common.serialization.StringSerializer");
        props.put("acks", "all");
        props.put("enable.idempotence", "true");
        this.dlq = new KafkaProducer<>(props);
        this.writer = Executors.newSingleThreadExecutor();
        this.permits = new Semaphore(MAX_INFLIGHT);
        this.inflight = new ArrayList<>();
    }
    @Override
    public void invoke(T value, Context context) {
        permits.acquireUninterruptibly();
        CompletableFuture<Void> f = CompletableFuture
            .runAsync(() -> writeWithRetry(value), writer)
            .whenComplete((r, e) -> permits.release());
        synchronized (inflight) {
            inflight.add(f);
            inflight.removeIf(CompletableFuture::isDone);
        }
    }
    // Checkpoint joins async writes; restore replays sources, MERGE keeps it idempotent.
    @Override
    public void snapshotState(FunctionSnapshotContext context) throws Exception {
        List<CompletableFuture<Void>> copy;
        synchronized (inflight) {
            copy = new ArrayList<>(inflight);
        }
        for (CompletableFuture<Void> f : copy) {
            f.get(30, TimeUnit.SECONDS);
        }
        synchronized (inflight) {
            inflight.removeIf(CompletableFuture::isDone);
        }
    }
    @Override
    public void initializeState(FunctionInitializationContext context) {
    }
    private void writeWithRetry(T value) {
        Exception last = null;
        for (int attempt = 0; attempt <= MAX_RETRIES; attempt++) {
            if (attempt > 0) {
                try {
                    Thread.sleep(BASE_BACKOFF_MS * (1L << (attempt - 1)));
                } catch (InterruptedException ie) {
                    Thread.currentThread().interrupt();
                    break;
                }
            }
            try (Session session = driver.session()) {
                Object[] kv = mapper.params(value).entrySet().stream()
                    .flatMap(e -> java.util.stream.Stream.of(e.getKey(), e.getValue()))
                    .toArray();
                session.run(mapper.cypher(), Values.parameters(kv)).consume();
                return;
            } catch (Exception e) {
                last = e;
            }
        }
        sendToDlq(value, last);
    }
    private void sendToDlq(T value, Exception cause) {
        try {
            String payload = value + "|error=" + (cause == null ? "unknown" : cause.getMessage());
            dlq.send(new ProducerRecord<>(dlqTopic, mapper.recordId(value), payload));
        } catch (Exception e) {
            throw new RuntimeException("DLQ send failed for record " + mapper.recordId(value), e);
        }
    }
    @Override
    public void close() {
        if (writer != null) {
            writer.shutdown();
        }
        try {
            if (driver != null) {
                driver.close();
            }
        } catch (Exception e) {
            throw new RuntimeException("Memgraph driver close failed", e);
        }
        try {
            if (dlq != null) {
                dlq.close();
            }
        } catch (Exception e) {
            throw new RuntimeException("DLQ producer close failed", e);
        }
    }
    public static MemgraphBoltSink<TemporalEdge> forTemporalEdges(String boltUri, String username, String password,
            String bootstrapServers, String dlqTopic) {
        CypherMapper<TemporalEdge> m = new CypherMapper<TemporalEdge>() {
            @Override
            public String cypher() {
                return "MERGE (s:Wallet {address: $src}) "
                    + "MERGE (t:Wallet {address: $dst}) "
                    + "MERGE (s)-[e:TRANSACTS {txHash: $tx}] "
                    + "ON CREATE SET e.validFrom = $from, e.validTo = $to";
            }

            @Override
            public Map<String, Object> params(TemporalEdge v) {
                Map<String, Object> p = new HashMap<>();
                p.put("src", v.getSource());
                p.put("dst", v.getTarget());
                p.put("tx", v.getTxHash());
                p.put("from", v.getValidFrom());
                p.put("to", v.getValidTo());
                return p;
            }

            @Override
            public String recordId(TemporalEdge v) {
                return v.getTxHash();
            }
        };
        return new MemgraphBoltSink<>(boltUri, username, password, bootstrapServers, dlqTopic, m);
    }
    public static MemgraphBoltSink<RankUpdate> forRankUpdates(String boltUri, String username, String password,
            String bootstrapServers, String dlqTopic) {
        CypherMapper<RankUpdate> m = new CypherMapper<RankUpdate>() {
            @Override
            public String cypher() {
                return "MERGE (n:Wallet {address: $node}) SET n.rank = $rank";
            }

            @Override
            public Map<String, Object> params(RankUpdate v) {
                Map<String, Object> p = new HashMap<>();
                p.put("node", v.getNode());
                p.put("rank", v.getRank());
                return p;
            }

            @Override
            public String recordId(RankUpdate v) {
                return v.getNode();
            }
        };
        return new MemgraphBoltSink<>(boltUri, username, password, bootstrapServers, dlqTopic, m);
    }
}
