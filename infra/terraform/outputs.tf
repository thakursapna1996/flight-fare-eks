output "cluster_name" {
  description = "EKS cluster name (use with: aws eks update-kubeconfig)"
  value       = module.eks.cluster_name
}

output "cluster_endpoint" {
  description = "EKS API server endpoint"
  value       = module.eks.cluster_endpoint
}

output "region" {
  value = var.aws_region
}

output "ecr_repository_url" {
  description = "Push images here"
  value       = aws_ecr_repository.app.repository_url
}

output "configure_kubectl" {
  description = "Run this to point kubectl at the new cluster"
  value       = "aws eks update-kubeconfig --region ${var.aws_region} --name ${module.eks.cluster_name}"
}
