# provider.tf
# ===========
# Configura Terraform y el provider de AWS apuntando al sandbox vía perfil SSO.

terraform {
  required_version = ">= 1.8"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region  = var.region
  profile = var.aws_profile # perfil SSO del sandbox (definido en terraform.tfvars)
}
