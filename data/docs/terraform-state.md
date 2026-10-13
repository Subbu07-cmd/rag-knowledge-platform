# Terraform State Management

## Remote state
Store Terraform state in an S3 bucket with versioning and encryption enabled, and use a DynamoDB table for state locking so that two engineers cannot apply at the same time.

## Drift
Run `terraform plan` on a schedule to detect drift between the real infrastructure and the code. Any drift should be fixed in code, never by hand in the console.

## Importing resources
Existing resources can be brought under management with an `import` block followed by `terraform plan` to confirm no changes.
