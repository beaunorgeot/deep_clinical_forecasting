# Deep Clinical Forecasting

The code within this repo provides a simple regime to build, train, and test the final deep learning model described in "Assessment of a Deep Learning Model Based on Electronic Health Record Data to Forecast Clinical Outcomes in Patients With Rheumatoid Arthritis.", Norgeot 2019. 


## Brief Abstract
We used structured Electronic Health Record (EHR) data, derived from rheumatology clinics at two distinct health systems (an academic health center and public safety net hospital) with different EHR platforms, to build a deep learning model that would predict future Rheumatoid Arthritis (RA) disease activity. Our results indicate that it is possible to build accurate models that generalize across hospitals with different EHR systems and diverse patient populations of only a few hundred patients.

Link to Manuscript on JAMA Network: https://jamanetwork.com/journals/jamanetworkopen/fullarticle/2728001

#### Full Abstract
**IMPORTANCE:**

Knowing the future condition of a patient would enable a physician to customize current therapeutic options to prevent disease worsening, but predicting that future condition requires sophisticated modeling and information. If artificial intelligence models were capable of forecasting future patient outcomes, they could be used to aid practitioners and patients in prognosticating outcomes or simulating potential outcomes under different treatment scenarios.

**OBJECTIVE:**

To assess the ability of an artificial intelligence system to prognosticate the state of disease activity of patients with rheumatoid arthritis (RA) at their next clinical visit.

**DESIGN, SETTING, AND PARTICIPANTS:**

This prognostic study included 820 patients with RA from rheumatology clinics at 2 distinct health care systems with different electronic health record platforms: a university hospital (UH) and a public safety-net hospital (SNH). The UH and SNH had substantially different patient populations and treatment patterns. The UH has records on approximately 1 million total patients starting in January 2012. The UH data for this study were accessed on July 1, 2017. The SNH has records on 65 000 unique individuals starting in January 2013. The SNH data for the study were collected on February 27, 2018.

**EXPOSURES:**

Structured data were extracted from the electronic health record, including exposures (medications), patient demographics, laboratories, and prior measures of disease activity. A longitudinal deep learning model was used to predict disease activity for patients with RA at their next rheumatology clinic visit and to evaluate interhospital performance and model interoperability strategies.

**MAIN OUTCOMES AND MEASURES:**

Model performance was quantified using the area under the receiver operating characteristic curve (AUROC). Disease activity in RA was measured using a composite index score.

**RESULTS:**

A total of 578 UH patients (mean [SD] age, 57 [15] years; 477 [82.5%] female; 296 [51.2%] white) and 242 SNH patients (mean [SD] age, 60 [15] years; 195 [80.6%] female; 30 [12.4%] white) were included in the study. Patients at the UH compared with those at the SNH were seen more frequently (median time between visits, 100 vs 180 days) and were more frequently prescribed higher-class medications (biologics) (364 [63.0%] vs 70 [28.9%]). At the UH, the model reached an AUROC of 0.91 (95% CI, 0.86-0.96) in a test cohort of 116 patients. The UH-trained model had an AUROC of 0.74 (95% CI, 0.65-0.83) in the SNH test cohort (n = 117) despite marked differences in the patient populations. In both settings, baseline prediction using each patients' most recent disease activity score had statistically random performance.

**CONCLUSIONS AND RELEVANCE:**

The findings suggest that building accurate models to forecast complex disease outcomes using electronic health record data is possible and these models can be shared across hospitals with diverse patient populations.