# Warsaw Clinical PI Outreach Kit & 15-Minute Demo Script

> **How to use this kit:** Pick 3–5 targets below, personalize the bracketed `[paper/topic]` line, attach `1_page_study_concept.md` (exported to PDF), and send from your university or personal email. **Never ask for hospital data in email #1** — ask for a 15-minute live demo and clinical feedback.

---

## 1. High-Priority Verified Warsaw Targets (2026)

### Target A — PTK Section of AI & Digital Solutions (SIRC) @ WIM-PIB Warsaw *(Top Priority!)*
- **Why this is a bullseye:** The Polish Cardiac Society's **Sekcja Sztucznej Inteligencji i Rozwiązań Cyfrowych (SIRC)** is headquartered in Warsaw at **Wojskowy Instytut Medyczny – Państwowy Instytut Badawczy (WIM-PIB)**, and their publicly stated #1 priority is **validating AI algorithms on local Polish hospital data** — literally what `heart-pipeline calibrate-site` solves!
- **Key Leadership:**
  - **prof. dr hab. n. med. Paweł Krzesiński** (Chair, SIRC PTK — WIM-PIB Warsaw)
  - **prof. dr hab. n. med. Piotr Szymański** (Chair-Elect, SIRC PTK — Warsaw)
- **Immediate In-Person Opportunity:** **I Konferencja Sekcji Sztucznej Inteligencji i Rozwiązań Cyfrowych PTK — 12 December 2026, Warsaw** (check `ptkardio.pl` for registration/student attendance — showing your live demo on a laptop during a coffee break is worth 50 cold emails!).

### Target B — 1st Chair & Department of Cardiology, Medical University of Warsaw (WUM)
- **Why they're a great fit:** Active publications on ESC AI/ML consensus statements, XGBoost cardiac risk stratification, AI algorithmic fairness, and Poland's first AI-equipped arrhythmia recorder implantations — plus active **Cardiology Student Research Circles (*SKN Kardiologii*)** open to CS collaborators.
- **How to pick the exact MD/PhD contact:** Run this PubMed query and email the **first author** (younger MD/PhD) or **corresponding author**:
  ```text
  ("Medical University of Warsaw"[Affiliation]) AND ("Cardiology"[Affiliation]) AND ("machine learning" OR "artificial intelligence")
  ```

### Target C — National Institute of Cardiology (NIKard, Warsaw-Anin)
- **Why they're a great fit:** Participated in EU CAD clinical decision-support consortia (e.g., *SMARTool*) and AI-Quantitative CT (AI-QCT) coronary plaque/CAD risk modeling.
- **PubMed query to find active junior investigators:**
  ```text
  ("National Institute of Cardiology"[Affiliation] AND "Warsaw"[Affiliation]) AND ("machine learning" OR "artificial intelligence" OR "coronary artery disease")
  ```

---

## 2. Ready-to-Send Outreach Email (English)

**Subject:** Warsaw CS student — offline clinical AI recalibration demo (15 min?)

Dear Prof. / Dr. [Last Name],

I am a third-year Computer Science student in Warsaw working on trustworthy clinical machine learning for cardiovascular risk triage, and I have been following [your work in SIRC PTK / your group's recent paper on ...].

Rather than claiming another black-box accuracy number, I built an open-source, offline 2-stage prototype (PyTorch Graph-Augmented Transformer + HL7 FHIR R4, zero-pickle, 127 CI tests) focused on **when risk models fail across hospitals** — specifically conformal abstention (`DEFER_TO_CLINICIAN`) and 2-second local hospital recalibration. On an unseen prospective Framingham cohort ($N=4{,}172$), zero-shot 90% coverage collapsed to 66.8% (`ECE 0.423`), and local recalibration restored it to **90.9%** (`ECE 0.0061`).

- Code & Model Cards: https://github.com/YahyaLendani/heart-disease-clinical-ml
- Portfolio: https://yahyalendani.me

I have attached a 1-page non-interventional silent-mode shadow study concept (zero patient risk, 100% on-premise Docker, zero PII). Would you have **15 minutes next week** (Zoom or coffee near [WIM / WUM / Anin]) for a brief live demo and your clinical critique?

Best regards,  
Yahya Lendani

---

## 3. Ready-to-Send Outreach Email (Polish)

**Temat:** Student informatyki (Warszawa) — krótka prezentacja narzędzia do lokalnej rekalibracji modeli AI w kardiologii (15 min?)

Szanowny Panie Profesorze / Szanowna Pani Doktor,

jestem studentem III roku informatyki w Warszawie i rozwijam projekt z zakresu wiarygodnego uczenia maszynowego (trustworthy clinical ML) w ocenie ryzyka sercowo-naczyniowego. Z dużym zainteresowaniem śledzę [działalność Sekcji SIRC PTK / ostatnią publikację Państwa zespołu dotyczącą ...].

Zamiast kolejnego „czarnoskrzynkowego” modelu skupiłem się na problemie **spadku jakości algorytmów po przeniesieniu do nowego szpitala** oraz bezpiecznym wdrażaniu w środowisku lokalnym. Zbudowałem otwartoźródłowy, w pełni offline'owy dwuetapowy prototyp (PyTorch Graph-Augmented Transformer + HL7 FHIR R4, 127 testów automatycznych), który wykrywa przypadki poza rozkładem (OOD), kieruje niepewne przypadki do weryfikacji lekarza (conformal triage) oraz umożliwia **2-sekundową lokalną rekalibrację** na danych szpitala (na zewnętrznej prospektywnej kohorcie Framingham $N=4172$ obniżyło to błąd kalibracji ECE o 98,6% i przywróciło pokrycie 90% z 66,8% do 90,9%).

- Repozytorium i karty modeli: https://github.com/YahyaLendani/heart-disease-clinical-ml
- Portfolio: https://yahyalendani.me

W załączniku przesyłam jednostronicowy zarys bezinwazyjnego badania obserwacyjnego w trybie *silent shadow mode* (zero ryzyka dla pacjenta, kontener Docker w sieci szpitala, brak PII). Czy znalazłby Pan Profesor / znalazłaby Pani Doktor **15 minut w przyszłym tygodniu** (online lub na miejscu) na krótkie demo i opinię kliniczną?

Z wyrazami szacunku,  
Yahya Lendani

---

## 4. 15-Minute Meeting Script
1. **Minutes 0–3 (Problem First):** Open `http://127.0.0.1:8000/slides` on **Slide 2** (LOCO hospital shift) and **Slide 6** (Framingham zero-shot collapse vs. 2-second `calibrate-site` recovery).
2. **Minutes 3–6 (Live Bedside Demo):** Open `http://127.0.0.1:8000`:
   - Click **Borderline (Defer)** → show conformal deferral band → click **Escalate to Stage-2 Cardiology Workup →**
   - Click **OOD Safety Alert** → show automatic fail-safe override
   - Click **Copy Clinical Note** & **Model Card** tab
3. **Minutes 6–15 (Listen & Co-Design):** Ask:
   > *"If you had a dedicated ML engineer for 3 months to run a silent observational study on retrospective or prospective data your clinic already collects, what clinical endpoint would be most valuable to validate and publish together?"*
4. **Close:** Offer to draft the *Komisja Bioetyczna* technical annex and hand over `hospital_it_gdpr_brief.md` for their DPO/IT review.
