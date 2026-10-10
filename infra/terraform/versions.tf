terraform {
  required_version = ">= 1.5"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.60"
    }
  }

  # For real use, store state remotely so it is shared and locked.
  # backend "s3" {
  #   bucket         = "my-tf-state-bucket"
  #   key            = "flight-fare/terraform.tfstate"
  #   region         = "eu-west-1"
  #   dynamodb_table = "terraform-locks"
  #   encrypt        = true
  # }
}
