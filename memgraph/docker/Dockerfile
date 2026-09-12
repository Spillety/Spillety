FROM memgraph/memgraph:latest

# Configure memory limits for 256GB RAM single node
ENV MEMGRAPH_RAM_LIMIT=268435456
ENV MEMGRAPH_WAL_ENABLED=true
ENV MEMGRAPH_WAL_SYNC_INTERVAL=1000

# Copy configuration
COPY memgraph.conf /etc/memgraph/memgraph.conf

EXPOSE 7687  # Bolt protocol
EXPOSE 3000  # Management API

CMD ["memgraph"]
