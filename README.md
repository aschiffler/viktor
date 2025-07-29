# Deployment

```
helm install my-rag ./ -n viktor --set lightrag.image.tag=1.1.0 --set neo4j.secret.neo4jPassword=myStrongPassword
```