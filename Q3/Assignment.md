    Question 3: Ethical Auditing &amp; &quot;Documentation Debt&quot; Mitigation

Objective: Perform a &quot;Sound Check&quot; audit on a public audio dataset to identify and correct
social biases.

Task:
Bias Identification: Select a public audio dataset and programmatically audit it for
&quot;Documentation Debt&quot; and representation bias (e.g., gender, age, or dialect).
The &quot;Privacy Preserving&quot; Transformation: Implement a &quot;Privacy Preserving AI&quot; module using
PyTorch that can obfuscate sensitive biometric traits (like changing a &quot;Male, Old&quot; voice to
&quot;Female, Young&quot;) while maintaining the linguistic content (ASR accuracy).
Fairness Loss Function: Modify the training loop of a speech recognition model by adding a
custom &quot;Fairness Loss&quot; term aimed at minimizing the performance gap between different
demographic groups identified in your audit.
Validation: Use DNSMOS or FAD (Frechet Audio Distance) to ensure that your privacy-preserving
transformations haven&#39;t introduced &quot;Toxicity Traps&quot; or significant audio artifacts that degrade
the system&#39;s &quot;Acceptability&quot;.

Deliverables: q3/ folder: audit.py, privacymodule.py, pp_demo.py, train_fair.py,
evaluation_scripts/(FAD/DNSMOS or proxies), examples/(audio pairs), audit_plots.pdf, and a
q3_report.pdf summarizing results and ethical considerations (max 4 pages)