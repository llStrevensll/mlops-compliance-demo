# main.tf
# =======
# Despliega el modelo en un endpoint SERVERLESS de SageMaker.
# Serverless = escala a cero, $0 en reposo, cobra solo por request.
# Recursos: S3 (modelo) + IAM (rol) + SageMaker Model + Endpoint Config + Endpoint.

# Identidad de la cuenta AWS activa (para el guard de seguridad).
data "aws_caller_identity" "current" {}

# Imagen del contenedor sklearn GESTIONADO por AWS (no usamos ECR propio).
data "aws_sagemaker_prebuilt_ecr_image" "sklearn" {
  repository_name = "sagemaker-scikit-learn"
  image_tag       = var.sklearn_image_tag
}

# --- Bucket S3 para el artefacto del modelo ---
resource "aws_s3_bucket" "model" {
  # Nombre único usando el account id (runtime, no queda en el repo).
  bucket        = "${var.project}-sagemaker-${data.aws_caller_identity.current.account_id}"
  force_destroy = true # permite que 'terraform destroy' borre el bucket aunque tenga objetos

  # GUARD DE SEGURIDAD: aborta si la cuenta activa NO es el sandbox aprobado.
  lifecycle {
    precondition {
      condition     = data.aws_caller_identity.current.account_id == var.sandbox_account_id
      error_message = "Cuenta AWS activa (${data.aws_caller_identity.current.account_id}) != sandbox esperado (${var.sandbox_account_id}). Abortando por seguridad."
    }
  }
}

# Sube el model.tar.gz al bucket.
resource "aws_s3_object" "model" {
  bucket = aws_s3_bucket.model.id
  key    = "model/model.tar.gz"
  source = "${path.module}/model.tar.gz"
  etag   = filemd5("${path.module}/model.tar.gz") # re-sube si el archivo cambia
}

# --- Rol IAM que SageMaker asume para ejecutar el modelo ---
resource "aws_iam_role" "sagemaker" {
  name = "${var.project}-sagemaker-exec"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "sagemaker.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

# Permisos: acceso completo de SageMaker (incluye leer buckets con 'sagemaker' en el nombre).
resource "aws_iam_role_policy_attachment" "sagemaker_full" {
  role       = aws_iam_role.sagemaker.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess"
}

# --- Modelo de SageMaker: une el contenedor sklearn + el artefacto del modelo ---
resource "aws_sagemaker_model" "this" {
  name               = "${var.project}-model"
  execution_role_arn = aws_iam_role.sagemaker.arn

  primary_container {
    image          = data.aws_sagemaker_prebuilt_ecr_image.sklearn.registry_path
    model_data_url = "s3://${aws_s3_bucket.model.id}/${aws_s3_object.model.key}"
    environment = {
      # Le dice al contenedor qué script usar y dónde está (dentro del tar, en code/).
      SAGEMAKER_PROGRAM           = "inference.py"
      SAGEMAKER_SUBMIT_DIRECTORY  = "/opt/ml/model/code"
    }
  }
}

# --- Configuración del endpoint: SERVERLESS (escala a cero) ---
resource "aws_sagemaker_endpoint_configuration" "this" {
  name = "${var.project}-endpoint-config"

  production_variants {
    variant_name = "AllTraffic"
    model_name   = aws_sagemaker_model.this.name

    serverless_config {
      memory_size_in_mb = var.serverless_memory_mb
      max_concurrency   = var.serverless_max_concurrency
    }
  }
}

# --- El endpoint en sí (lo que invocamos para predecir) ---
resource "aws_sagemaker_endpoint" "this" {
  name                 = "${var.project}-endpoint"
  endpoint_config_name = aws_sagemaker_endpoint_configuration.this.name
}
