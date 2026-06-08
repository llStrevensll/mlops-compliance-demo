# variables.tf
# ============
# Variables del módulo. Las SENSIBLES (perfil, account id) NO tienen default
# y se pasan por terraform.tfvars (gitignored) -> nunca llegan al repo público.

variable "aws_profile" {
  description = "Perfil de AWS CLI (SSO) del sandbox. Se define en terraform.tfvars."
  type        = string
}

variable "sandbox_account_id" {
  description = "Account ID del sandbox aprobado. Guard de seguridad (ver main.tf)."
  type        = string
}

variable "region" {
  description = "Región de AWS."
  type        = string
  default     = "us-east-1"
}

variable "project" {
  description = "Prefijo para nombrar recursos."
  type        = string
  default     = "compliance-ml"
}

variable "sklearn_image_tag" {
  description = "Tag del contenedor sklearn gestionado de AWS."
  type        = string
  default     = "1.2-1"
}

variable "serverless_memory_mb" {
  description = "Memoria del endpoint serverless (MB)."
  type        = number
  default     = 2048
}

variable "serverless_max_concurrency" {
  description = "Concurrencia máxima del endpoint serverless."
  type        = number
  default     = 1
}
