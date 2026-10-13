# Kubernetes Rollbacks

## Rolling back a deployment
To roll back a Kubernetes deployment to the previous revision, run `kubectl rollout undo deployment/<name>`.
To roll back to a specific revision, list the history with `kubectl rollout history deployment/<name>` and then run `kubectl rollout undo deployment/<name> --to-revision=<n>`.

## GitOps rollbacks
When Argo CD manages the deployment, revert the commit in Git instead of running kubectl. Argo CD self-heal would otherwise re-apply the broken version.

## Verifying
Watch the rollout with `kubectl rollout status deployment/<name>` and confirm the error rate returns below the SLO threshold.
