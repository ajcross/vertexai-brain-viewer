# Web frontend for BrAIn decoder model on Vertex AI 

This is a simple frontend to display images generated by [BrAIn](https://github.com/mtablado/uoc2022_tfm) decoder model running on a VertexAI endpoint

Can be deployed on a VM, container, cloudrun or any google cloud environment with a service account attached with correct permission

## Configuration

It uses runtime service account for authentication. Permission `aiplatform.endpoints.predict` is required. This permission is included in Vertex AI User (`roles/aiplatform.user`), although it is a good idea to create a custom role to limit permission.

The following environment variables must be configured with the correct values

- `ENDPOINT_ID`
- `PROJECT_ID`
- `REGION`

Optionally, it is possible to configure the title html element with the environment variable `TITLE`
