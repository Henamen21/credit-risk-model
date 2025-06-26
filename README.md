# Credit Scoring Model for Bati Bank

### Objectives

- ✅ Define a **proxy variable** to categorize users as **high risk** (bad) or **low risk** (good).
- ✅ Select **observable features** that are strong predictors of default.
- ✅ Develop a model that estimates **risk probability** for a new customer.
- ✅ Develop a model that maps **risk probability** to a **credit score**.
- ✅ Build a model that predicts the **optimal loan amount and duration**.

---

## Credit Scoring Business Understanding

### 1. How does the Basel II Accord’s emphasis on risk measurement influence our need for an interpretable and well-documented model?

**Basel II** emphasizes quantitative risk measurement and regulatory compliance in credit risk modeling. This mandates that any credit scoring model must be:

- 🔍 **Transparent** and **interpretable** to auditors and regulators.
- 📄 **Well-documented**, clearly outlining variable derivation and model logic.
- 💬 **Explainable**, so decisions on approvals or rejections can be justified.

In this project, we follow these guidelines by using **logistic regression** and **feature engineering via the RFMS framework**. Each variable (e.g., recency or monetary value of consumption) is tied to meaningful business behavior, ensuring the model is interpretable and compliant with Basel II.

---

### 2. Since we lack a direct "default" label, why is creating a proxy variable necessary, and what are the potential business risks of making predictions based on this proxy?

While this dataset includes a "default" label (`Yi ∈ {0,1}`), in many real-world cases, **true defaults** are:

- Delayed or occur long after early risk signals.
- Unavailable due to **privacy regulations**.
- Not timely for **early intervention**.

#### Why we use proxies:

- 🕐 **Lags in default data** make early risk prediction hard.
- 🛡️ **Privacy laws** limit access to financial histories.
- 🔔 Companies need **early warning systems** to intervene proactively.

#### Business Risks of Using Proxies:

- ❌ **False Positives**: Good applicants rejected — leads to lost revenue.
- ⚠️ **False Negatives**: Risky applicants approved — leads to default losses.
- 🔁 **Model Drift**: Proxies may degrade over time if behaviors change.
- ⚖️ **Regulatory Risk**: Using opaque proxies may violate **fair lending** laws.

📌 Therefore, **true default labels** (when available) are always preferable.

---

### 3. What are the key trade-offs between using a simple, interpretable model (like Logistic Regression with WoE) versus a complex, high-performance model (like Gradient Boosting) in a regulated financial context?

| Aspect              | Logistic Regression (Simple)                        | Gradient Boosting (Complex)                      |
|---------------------|-----------------------------------------------------|--------------------------------------------------|
| **Interpretability** | ✅ High – easy to explain coefficients               | ❌ Low – often treated as a black box             |
| **Compliance**       | ✅ Easily satisfies Basel II/III                    | ⚠️ Harder to justify decisions                    |
| **Transparency**     | ✅ Fully auditable and easy to document             | ⚠️ Requires explainability tools (e.g., SHAP)     |
| **Performance**      | 👍 Good if features are well-engineered             | 🚀 Higher accuracy with non-linear relationships  |
| **Business Trust**   | ✅ More trusted by stakeholders                     | ⚠️ Can face skepticism due to complexity          |
| **Maintenance**      | ✅ Easier to maintain and retrain                   | ⚠️ More resource-intensive                        |

In regulated environments, **simplicity and auditability** often take precedence over marginal gains in accuracy. Therefore, models like **logistic regression with WoE or RFMS** features are preferred for their **regulatory robustness and explainability**.

---
