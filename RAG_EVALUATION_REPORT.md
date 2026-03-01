# Afghan Legal RAG Evaluation Report

## 📊 Summary
- **Total Questions**: 30
- **Correct (Source Found)**: 25
- **Incorrect (Source Missed)**: 5
- **Accuracy**: 83.3%

---

## ✅ Passed Questions
- **Q**: What is the legal age for marriage for boys and girls?
  - **Source**: Article 70
  - **Status**: PASSED

- **Q**: Under what conditions is a man allowed to marry more than one wife?
  - **Source**: Article 86
  - **Status**: PASSED

- **Q**: Until what age does a mother keep custody of her son and daughter?
  - **Source**: Article 249
  - **Status**: PASSED

- **Q**: How much inheritance does a husband get if his wife dies and they have children?
  - **Source**: Article 2007
  - **Status**: PASSED

- **Q**: What is the maximum period for a lease if the person administering the property is not the owner (e.g. a guardian)?
  - **Source**: Article 1325
  - **Status**: PASSED

- **Q**: Does a partnership end if one of the partners dies?
  - **Source**: Article 1250
  - **Status**: PASSED

- **Q**: What is the right of Preemption (Shufa)?
  - **Source**: Article 2213
  - **Status**: PASSED

- **Q**: Is preemption allowed if the property was given as a gift or inheritance?
  - **Source**: Article 2226
  - **Status**: PASSED

- **Q**: If a man and woman are engaged and they exchange gifts, but then one of them cancels the engagement, can they get the gifts back?
  - **Source**: Article 65
  - **Status**: PASSED

- **Q**: If a wife donates her dowry (Mahr) to her husband but they get divorced before the marriage is consummated, can the husband claim half of it back?
  - **Source**: Article 111
  - **Status**: PASSED

- **Q**: If a divorced woman is pregnant, when does her waiting period (Iddah) end?
  - **Source**: Article 206
  - **Status**: PASSED

- **Q**: What happens to the right of custody if the mother marries a stranger (someone not related to the child)?
  - **Source**: Article 245
  - **Status**: PASSED

- **Q**: How many years must a person possess a piece of land to claim ownership of it through 'Lapse of Time' (Adverse Possession)?
  - **Source**: Article 2279
  - **Status**: PASSED

- **Q**: If someone finds buried ancient relics or treasure on their own private land, who owns it?
  - **Source**: Article 1988
  - **Status**: PASSED

- **Q**: Can a person own public property, like a bridge or public park, by possessing it for a long time?
  - **Source**: Article 482
  - **Status**: PASSED

- **Q**: If a person plants crops on another person's land without permission, who do the crops belong to?
  - **Source**: Article 2208
  - **Status**: PASSED

- **Q**: If I own the upper floor of a building and you own the lower floor, do I have the right to live on the roof of your floor?
  - **Source**: Article 1979
  - **Status**: PASSED

- **Q**: In a sale contract, if the buyer buys something without seeing it (Option of Sight), when can he reject it?
  - **Source**: Article 1044
  - **Status**: PASSED

- **Q**: What is the limitation period for filing a lawsuit regarding a defect in a purchased item?
  - **Source**: Article 1107
  - **Status**: PASSED

- **Q**: If an architect designs a building plan but does not supervise the construction, is he liable for defects in the building?
  - **Source**: Article 1492
  - **Status**: PASSED

- **Q**: If a person hires a worker (Labour Contract) but does not specify the wage, how is the wage determined?
  - **Source**: Article 1535
  - **Status**: PASSED

- **Q**: If a person intentionally kills their father, can they inherit from him?
  - **Source**: Article 1999
  - **Status**: PASSED

- **Q**: How is the inheritance divided if the deceased leaves behind a daughter and no sons?
  - **Source**: Article 2008
  - **Status**: PASSED

- **Q**: Who has a stronger right to Preemption (Shufa): a partner in the property or a neighbor?
  - **Source**: Article 2221
  - **Status**: PASSED

- **Q**: Can a neighbor open a window looking directly into my house if the distance is less than one meter?
  - **Source**: Article 1929
  - **Status**: PASSED

---

## ❌ Failed Questions & Diagnosis
### Q: Can a person make a will for their entire property to go to a stranger?
- **Expected Source**: Article 2137
- **AI Answer**: Based on the provided legal articles, a person cannot make a will for their entire property to go to a stranger. Article 2131 states that if a will is made for specific persons, the share of those persons for whom a will may not be made shall be returned to the inheritance of the testator. This impl...
- **Top Ranked Article ID**: 275
- **Diagnosis**: Re-ranking Error. The article 2137 was likely retrieved (found top article 275) but was pushed out of the top selection or the AI failed to cite it correctly.

### Q: If I buy a stolen item from a market in good faith, do I have to give it back to the original owner?
- **Expected Source**: Article 2290
- **AI Answer**: According to Article 1170, if you obtained the item by paying consideration and in good faith, you acquire a real right on the object of sale, and the provisions that would normally require returning it to the original owner do not apply. This means you are not obligated to return the stolen item to...
- **Top Ranked Article ID**: 1154
- **Diagnosis**: Re-ranking Error. The article 2290 was likely retrieved (found top article 1154) but was pushed out of the top selection or the AI failed to cite it correctly.

### Q: Can a mother demand payment (wages) for breastfeeding her own child?
- **Expected Source**: Article 230
- **AI Answer**: Yes, a mother can demand payment for breastfeeding her own child, but only under certain conditions.

According to Article 232, a mother is entitled to remuneration for giving milk to her child for up to two years. After two years, she is not entitled to any remuneration.

However, Article 233 state...
- **Top Ranked Article ID**: 229
- **Diagnosis**: Re-ranking Error. The article 230 was likely retrieved (found top article 229) but was pushed out of the top selection or the AI failed to cite it correctly.

### Q: Can an agent appointed to buy a specific house buy it for himself instead?
- **Expected Source**: Article 1583
- **AI Answer**: Article 1158 states that a broker or expert to whom properties are deposited for sale or price determination may not buy them for himself under his name or a pseudonym. This rule applies to agents appointed to purchase specific properties. Therefore, an agent appointed to buy a specific house cannot...
- **Top Ranked Article ID**: 1581
- **Diagnosis**: Re-ranking Error. The article 1583 was likely retrieved (found top article 1581) but was pushed out of the top selection or the AI failed to cite it correctly.

### Q: If two people die in the same accident (e.g., a fire or drowning) and we don't know who died first, do they inherit from each other?
- **Expected Source**: Article 1996
- **AI Answer**: According to Article 2059, if two people die in the same incident, such as drowning or fire, and it is not known which one died first, they shall not inherit from one another. Their shares will be divided among their living heirs....
- **Top Ranked Article ID**: 2059
- **Diagnosis**: Re-ranking Error. The article 1996 was likely retrieved (found top article 2059) but was pushed out of the top selection or the AI failed to cite it correctly.

