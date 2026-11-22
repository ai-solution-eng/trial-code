# Login to podman

podman login https://harbor.ai-application.pcai0104.ld7.hpecolo.net/

# Build the image locally

podman build --tag harbor.ai-application.pcai0104.ld7.hpecolo.net/rag/demo:0.0.1 .

# Push the image to Harbor

podman push harbor.ai-application.pcai0104.ld7.hpecolo.net/rag/demo:0.0.1
