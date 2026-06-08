# outputs.tf
# ==========
# Valores útiles que Terraform muestra al terminar el apply.

output "endpoint_name" {
  description = "Nombre del endpoint de SageMaker (para invocarlo)."
  value       = aws_sagemaker_endpoint.this.name
}

output "region" {
  description = "Región donde se desplegó."
  value       = var.region
}

output "model_bucket" {
  description = "Bucket S3 con el artefacto del modelo."
  value       = aws_s3_bucket.model.id
}
