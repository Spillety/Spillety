protoc --python_out=kafka/src --proto_path=kafka/proto kafka/proto/events.proto
protoc --python_out=kafka/src --proto_path=kafka/proto kafka/proto/enrichment.proto
