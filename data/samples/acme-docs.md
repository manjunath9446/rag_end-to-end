# Acme Cloud — Product Documentation

## Plans

### Starter
- Price: $0 per month
- Events: up to 1 million per month
- Retention: 30 days
- Support: community forum only

### Growth
- Price: $299 per month
- Events: up to 50 million per month
- Retention: 12 months
- Support: email with a 24-hour response target
- Overage: $0.40 per additional 10,000 events

### Enterprise
- Price: custom
- Events: unlimited under a fair-use policy
- Retention: up to 7 years
- Support: dedicated Slack channel, 1-hour P1 response SLA
- Includes SSO, SCIM, and audit logs

## Security

- Data encrypted in transit with TLS 1.2 or higher
- Data encrypted at rest with AES-256
- Encryption keys rotated every 90 days
- SOC 2 Type II, ISO 27001, and GDPR compliant
- HIPAA available on Enterprise with a signed BAA

## API

- Authentication: Bearer token in the Authorization header
- Rate limits: 100 requests per second per project on Growth; Enterprise is negotiated
- Rotated keys remain valid for 24 hours to allow zero-downtime rollout

## Data Retention

- Starter: 30 days
- Growth: 12 months
- Enterprise: custom, up to 7 years
- Deleted data removed from hot storage immediately, from backups within 30 days