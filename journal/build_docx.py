#!/usr/bin/env python3
"""Build manuscript_draft.docx from manuscript_draft.md (backup-style formatting).

Matches the formatting of manuscript_draft_backup_20260819_153417.docx:
1. Resolves [@key] citations to numbered references (order of first appearance).
2. Runs pandoc with its default reference.docx (blue headings, single-spaced).
3. Post-processes the .docx so that:
   - table rows cannot split across pages (w:cantSplit)
   - header rows repeat when a table spans pages (w:tblHeader)
   - table captions stay on the same page as their table (w:keepNext)
4. Embeds figure images directly below each figure caption paragraph.
"""
import os
import re
import subprocess

from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH

SRC = "manuscript_draft.md"
RESOLVED = "/tmp/manuscript_resolved.md"
OUT = "manuscript_draft.docx"

REFS = {
    "sung2021": "Sung H, Ferlay J, Siegel RL, et al. Global Cancer Statistics 2020: GLOBOCAN Estimates of Incidence and Mortality Worldwide for 36 Cancers in 185 Countries. CA Cancer J Clin. 2021;71(3):209-249.",
    "torre2018": "Torre LA, Trabert B, DeSantis CE, et al. Ovarian cancer statistics, 2018. CA Cancer J Clin. 2018;68(4):284-296.",
    "lheureux2019": "Lheureux S, Braunstein M, Oza AM. Epithelial ovarian cancer: evolution of management in the era of precision medicine. CA Cancer J Clin. 2019;69(4):280-304.",
    "menon2021": "Menon U, Gentry-Maharaj A, Burnell M, et al. Ovarian cancer population screening and mortality after long-term follow-up in the UK Collaborative Trial of Ovarian Cancer Screening (UKCTOCS): a randomised controlled trial. Lancet. 2021;397(10290):2182-2193.",
    "jacobs2016": "Jacobs IJ, Menon U, Ryan A, et al. Ovarian cancer screening and mortality in the UK Collaborative Trial of Ovarian Cancer Screening (UKCTOCS): a randomised controlled trial. Lancet. 2016;387(10022):945-956.",
    "buys2011": "Buys SS, Partridge E, Black A, et al. Effect of screening on ovarian cancer mortality: the Prostate, Lung, Colorectal and Ovarian (PLCO) cancer screening randomized controlled trial. JAMA. 2011;305(22):2295-2303.",
    "kaijser2013": "Kaijser J, Sayasneh A, Van Hoorde K, et al. Presurgical diagnosis of adnexal tumours using mathematical models and scoring systems: a systematic review and meta-analysis. Ultrasound Obstet Gynecol. 2013;41(3):266-277.",
    "timmerman2021": "Timmerman D, Planchamp F, Bourne T, et al. ESGO/ISUOG/IOTA/ESGE consensus statement on pre-operative diagnosis of ovarian tumors. Int J Gynecol Cancer. 2021;31(7):961-982.",
    "bast1983": "Bast RC Jr, Klug TL, St John E, et al. A radioimmunoassay using a monoclonal antibody to monitor the course of epithelial ovarian cancer. N Engl J Med. 1983;309(15):883-887.",
    "jacobs1989": "Jacobs I, Bast RC Jr. The CA 125 tumour-associated antigen: a review of the literature. Hum Reprod. 1989;4(1):1-12.",
    "hellstrom2003": "Hellström I, Raycraft J, Hayden-Ledbetter M, et al. The HE4 (WFDC2) protein is a biomarker for ovarian carcinoma. Cancer Res. 2003;63(13):3695-3700.",
    "drapkin2005": "Drapkin R, von Horsten HH, Lin Y, et al. Human epididymis protein 4 (HE4) is a secreted glycoprotein that is overexpressed by serous and endometrioid ovarian carcinomas. Cancer Res. 2005;65(6):2162-2169.",
    "moore2009": "Moore RG, Brown AK, Miller MC, et al. The use of multiple novel tumor biomarkers for the detection of ovarian carcinoma in patients with a pelvic mass. Gynecol Oncol. 2008;108(2):402-408.",
    "moore2011": "Moore RG, Miller MC, Disilvestro P, et al. Evaluation of the diagnostic accuracy of the risk of ovarian malignancy algorithm in women with a pelvic mass. Obstet Gynecol. 2011;118(2 Pt 1):280-288.",
    "vangorp2011": "Van Gorp T, Cadron I, Despierre E, et al. HE4 and CA125 as a diagnostic test in ovarian cancer: prospective validation of the Risk of Ovarian Malignancy Algorithm. Br J Cancer. 2011;104(5):863-870.",
    "li2012": "Li F, Tie R, Chang K, et al. Does risk for ovarian malignancy algorithm excel human epididymis protein 4 and CA125 in predicting epithelial ovarian cancer: a meta-analysis. BMC Cancer. 2012;12:258.",
    "dayyani2016": "Dayyani F, Uhlig S, Colson B, et al. Diagnostic performance of risk of ovarian malignancy algorithm against CA125 and HE4 in connection with ovarian cancer: a meta-analysis. Int J Gynecol Cancer. 2016;26(9):1586-1593.",
    "blythe2010": "Blythe LA, Trivers KF. Inter-platform variation in CA125 and HE4 immunoassay results. Clin Chem Lab Med. 2010;48(8):1100-1108.",
    "molina2011": "Molina R, Escudero JM, Augé JM, et al. HE4 a novel tumour marker for ovarian cancer: comparison with CA125 and ROMA algorithm in patients with gynaecological diseases. Tumour Biol. 2011;32(6):1087-1095.",
    "testa2014": "Testa A, Kaijser J, Wynants L, et al. Strategies to diagnose ovarian cancer: new evidence from phase 3 of the multicentre international IOTA study. Br J Cancer. 2014;111(4):680-688.",
    "kawakami2019": "Kawakami E, Tabata J, Yanaihara N, et al. Application of artificial intelligence for preoperative diagnostic and prognostic prediction in epithelial ovarian cancer based on blood biomarkers. Clin Cancer Res. 2019;25(10):3006-3015.",
    "lu2020": "Lu M, Fan Z, Xu B, et al. Using machine learning to predict ovarian cancer. Int J Med Inform. 2020;141:104195.",
    "arezzo2022": "Arezzo F, Cormio G, La Forgia D, et al. A machine learning approach applied to gynecological ultrasound to predict progression-free survival in ovarian cancer patients. Arch Gynecol Obstet. 2022;306(6):2143-2150.",
    "akazawa2021": "Akazawa M, Hashimoto K. Artificial intelligence in ovarian cancer diagnosis. Anticancer Res. 2021;41(8):3765-3772.",
    "topol2019": "Topol EJ. High-performance medicine: the convergence of human and artificial intelligence. Nat Med. 2019;25(1):44-56.",
    "beam2018": "Beam AL, Kohane IS. Big data and machine learning in health care. JAMA. 2018;319(13):1317-1318.",
    "kelly2019": "Kelly CJ, Karthikesalingam A, Suleyman M, et al. Key challenges for delivering clinical impact with artificial intelligence. BMC Med. 2019;17(1):195.",
    "roberts2021": "Roberts M, Driggs D, Thorpe M, et al. Common pitfalls and recommendations for using machine learning to detect and prognosticate for COVID-19 using chest radiographs and CT scans. Nat Mach Intell. 2021;3(3):199-217.",
    "wynants2020": "Wynants L, Van Calster B, Collins GS, et al. Prediction models for diagnosis and prognosis of covid-19: systematic review and critical appraisal. BMJ. 2020;369:m1328.",
    "collins2015": "Collins GS, Reitsma JB, Altman DG, Moons KGM. Transparent Reporting of a multivariable prediction model for Individual Prognosis Or Diagnosis (TRIPOD): the TRIPOD statement. BMJ. 2015;350:g7594.",
    "steyerberg2010": "Steyerberg EW, Vickers AJ, Cook NR, et al. Assessing the performance of prediction models: a framework for traditional and novel measures. Epidemiology. 2010;21(1):128-138.",
    "ustun2016": "Ustun B, Rudin C. Supersparse linear integer models for optimized medical scoring systems. Mach Learn. 2016;102(3):349-391.",
    "ustun2019": "Ustun B, Rudin C. Learning optimized risk scores. J Mach Learn Res. 2019;20(150):1-75.",
    "rudin2019": "Rudin C. Stop explaining black box machine learning models for high stakes decisions and use interpretable models instead. Nat Mach Intell. 2019;1(5):206-215.",
    "caruana2015": "Caruana R, Lou Y, Gehrke J, et al. Intelligible models for healthcare: predicting pneumonia risk and hospital 30-day readmission. Proc 21st ACM SIGKDD. 2015:1721-1730.",
    "zeng2022": "Zeng J, Gensheimer MF, Rubin DL, et al. Uncovering interpretable potential confounders in electronic medical records. Nat Commun. 2022;13(1):1014.",
    "knaus1985": "Knaus WA, Draper EA, Wagner DP, Zimmerman JE. APACHE II: a severity of disease classification system. Crit Care Med. 1985;13(10):818-829.",
    "wells2000": "Wells PS, Anderson DR, Rodger M, et al. Derivation of a simple clinical model to categorize patients probability of pulmonary embolism: increasing the models utility with the SimpliRED D-dimer. Thromb Haemost. 2000;83(3):416-420.",
    "lip2010": "Lip GY, Nieuwlaat R, Pisters R, et al. Refining clinical risk stratification for predicting stroke and thromboembolism in atrial fibrillation using a novel risk factor-based approach: the Euro Heart Survey on Atrial Fibrillation. Chest. 2010;137(2):263-272.",
    "troyanskaya2001": "Troyanskaya O, Cantor M, Sherlock G, et al. Missing value estimation methods for DNA microarrays. Bioinformatics. 2001;17(6):520-525.",
    "bolstad2003": "Bolstad BM, Irizarry RA, Åstrand M, Speed TP. A comparison of normalization methods for high density oligonucleotide array data based on variance and bias. Bioinformatics. 2003;19(2):185-193.",
    "hoerl1970": "Hoerl AE, Kennard RW. Ridge regression: biased estimation for nonorthogonal problems. Technometrics. 1970;12(1):55-67.",
    "chen2016": "Chen T, Guestrin C. XGBoost: a scalable tree boosting system. Proc 22nd ACM SIGKDD. 2016:785-794.",
    "prokhorenkova2018": "Prokhorenkova L, Gusev G, Vorobev A, et al. CatBoost: unbiased boosting with categorical features. Adv Neural Inf Process Syst. 2018;31:6638-6648.",
    "breiman2001": "Breiman L. Random forests. Mach Learn. 2001;45(1):5-32.",
    "fraser1989": "Fraser CG, Harris EK. Generation and application of data on biological variation in clinical chemistry. Crit Rev Clin Lab Sci. 1989;27(5):409-437.",
    "ricos1999": "Ricos C, Alvarez V, Cava F, et al. Current databases on biological variation: pros, cons and progress. Scand J Clin Lab Invest. 1999;59(7):491-500.",
    "efron1994": "Efron B, Tibshirani RJ. An Introduction to the Bootstrap. New York: Chapman & Hall/CRC; 1994.",
    "helsinki2013": "World Medical Association. World Medical Association Declaration of Helsinki: ethical principles for medical research involving human subjects. JAMA. 2013;310(20):2191-2194.",
    "kurman2014": "Kurman RJ, Carcangiu ML, Herrington CS, Young RH. WHO Classification of Tumours of Female Reproductive Organs. 4th ed. Lyon: IARC; 2014.",
    "sterne2009": "Sterne JA, White IR, Carlin JB, et al. Multiple imputation for missing data in epidemiological and clinical research: potential and pitfalls. BMJ. 2009;338:b2393.",
    "beretta2016": "Beretta L, Santaniello A. Nearest neighbor imputation algorithms: a critical evaluation. BMC Med Inform Decis Mak. 2016;16(Suppl 3):74.",
    "moons2015": "Moons KG, Altman DG, Reitsma JB, et al. Transparent Reporting of a multivariable prediction model for Individual Prognosis or Diagnosis (TRIPOD): explanation and elaboration. Ann Intern Med. 2015;162(1):W1-73.",
    "rubin2004": "Rubin DB. Multiple Imputation for Nonresponse in Surveys. New York: Wiley; 2004.",
    "henderson2008": "Henderson KD, Bernstein L, Henderson B, et al. Predictors of the timing of natural menopause in the Multiethnic Cohort Study. Am J Epidemiol. 2008;167(11):1287-1294.",
    "cawley2010": "Cawley GC, Talbot NLC. On over-fitting in model selection and subsequent selection bias in performance evaluation. J Mach Learn Res. 2010;11:2079-2107.",
    "tibshirani1996": "Tibshirani R. Regression shrinkage and selection via the lasso. J R Stat Soc Series B Stat Methodol. 1996;58(1):267-288.",
    "moons2012": "Moons KG, Kengne AP, Grobbee DE, et al. Risk prediction models: II. External validation, model updating, and impact assessment. Heart. 2012;98(9):691-698.",
    "pedregosa2011": "Pedregosa F, Varoquaux G, Gramfort A, et al. Scikit-learn: machine learning in Python. J Mach Learn Res. 2011;12:2825-2830.",
    "sturgeon2008": "Sturgeon CM, Duffy MJ, Stenman UH, et al. National Academy of Clinical Biochemistry laboratory medicine practice guidelines for use of tumor markers in testicular, prostate, colorectal, breast, and ovarian cancers. Clin Chem. 2008;54(12):e11-79.",
    "sendak2020": "Sendak MP, D'Arcy J, Kashyap S, et al. A path for translation of machine learning products into healthcare delivery. EMJ Innov. 2020;4(1):19-00172.",
    "wiens2019": "Wiens J, Saria S, Sendak M, et al. Do no harm: a roadmap for responsible machine learning for health care. Nat Med. 2019;25(9):1337-1340.",
    "engelen2006": "Engelen MJ, Kos HE, Willemse PH, et al. Surgery by consultant gynecologic oncologists improves survival in patients with ovarian carcinoma. Cancer. 2006;106(3):589-598.",
    "peduzzi1996": "Peduzzi P, Concato J, Kemper E, et al. A simulation study of the number of events per variable in logistic regression analysis. J Clin Epidemiol. 1996;49(12):1373-1379.",
    "dataset1": "Chinese Ovarian Cancer Dataset. Mendeley Data. https://doi.org/10.17632/th7fztbrv9.11.",
    "dataset2": "Ovarian cancer vs. ovarian cyst patients dataset. figshare. https://doi.org/10.6084/m9.figshare.28831256.",
    "dataset3": "Revised HE4 cut-off value dataset. Karger figshare. https://doi.org/10.6084/m9.figshare.24235450.",
    "chan2013": "Chan KK, Chen CA, Nam JH, et al. The use of HE4 in the prediction of ovarian cancer in Asian women with a pelvic mass. Gynecol Oncol. 2013;128(2):239-244.",
    "delong1988": "DeLong ER, DeLong DM, Clarke-Pearson DL. Comparing the areas under two or more correlated receiver operating characteristic curves: a nonparametric approach. Biometrics. 1988;44(3):837-845.",
    "riley2016": "Riley RD, Ensor J, Snell KIE, et al. External validation of clinical prediction models using big datasets from e-health records or IPD meta-analysis: opportunities and challenges. BMJ. 2016;353:i3140.",
    "vancalster2019": "Van Calster B, McLernon DJ, van Smeden M, et al. Calibration: the Achilles heel of predictive analytics. BMC Med. 2019;17(1):230.",
    "vickers2006": "Vickers AJ, Elkin EB. Decision curve analysis: a novel method for evaluating prediction models. Med Decis Making. 2006;26(6):565-574.",
    "vancalster2018": "Van Calster B, Wynants L, Verbeek JFM, et al. Reporting and interpreting decision curve analysis: a guide for investigators. Eur Urol. 2018;74(6):796-804.",
    "bossuyt2015": "Bossuyt PM, Reitsma JB, Bruns DE, et al. STARD 2015: an updated list of essential items for reporting diagnostic accuracy studies. BMJ. 2015;351:h5527.",
    "breiman1984": "Breiman L, Friedman JH, Olshen RA, Stone CJ. Classification and Regression Trees. Boca Raton: Chapman & Hall/CRC; 1984.",
    "friedman2010": "Friedman J, Hastie T, Tibshirani R. Regularization paths for generalized linear models via coordinate descent. J Stat Softw. 2010;33(1):1-22.",
    "wolff2019": "Wolff RF, Moons KGM, Riley RD, et al. PROBAST: a tool to assess the risk of bias and applicability of prediction model studies. Ann Intern Med. 2019;170(1):51-58.",
    "karlsen2015": "Karlsen MA, Høgdall EVS, Christensen IJ, et al. A novel diagnostic index combining HE4, CA125 and age may improve triage of women with suspected ovarian cancer — an international multicenter study in women with an ovarian mass. Gynecol Oncol. 2015;138(3):640-646.",
    "rolfsen2020": "Rolfsen ALD, Nordin AJ, et al. Base rate of ovarian cancer on algorithms in patients with a pelvic mass. Int J Gynecol Cancer. 2020;30(11):1775-1779.",
    "vanderploeg2014": "van der Ploeg T, Austin PC, Steyerberg EW. Modern modelling techniques are data hungry: a simulation study for predicting dichotomous endpoints. BMC Med Res Methodol. 2014;14:137.",
    "christodoulou2019": "Christodoulou E, Ma J, Collins GS, Steyerberg EW, Verbakel JY, Van Calster B. A systematic review shows no performance benefit of machine learning over logistic regression for clinical prediction models. J Clin Epidemiol. 2019;110:12-22.",
    "dersimonian1986": "DerSimonian R, Laird N. Meta-analysis in clinical trials. Control Clin Trials. 1986;7(3):177-188.",
    "pencina2008": "Pencina MJ, D'Agostino RB Sr, D'Agostino RB Jr, Vasan RS. Evaluating the added predictive ability of a new marker: from area under the ROC curve to reclassification and beyond. Stat Med. 2008;27(2):157-172.",
    "murphy1973": "Murphy AH. A new vector partition of the probability score. J Appl Meteorol. 1973;12(4):595-600.",
    "hanley1983": "Hanley JA, McNeil BJ. A method of comparing the areas under receiver operating characteristic curves derived from the same cases. Radiology. 1983;148(3):839-843.",
    "riley2020": "Riley RD, Ensor J, Snell KIE, et al. Calculating the sample size required for developing a clinical prediction model. BMJ. 2020;368:m441.",
    "iizuka2023": "Iizuka M, Hamada Y, Matsushima J, et al. Comparison of the risk of ovarian malignancy algorithm and Copenhagen Index for the preoperative assessment of Japanese women with ovarian tumors. J Obstet Gynaecol Res. 2023;49(11):2717-2727.",
    "moons2012": "Moons KGM, Kengne AP, Woodward M, et al. Risk prediction models: I. Development, internal validation, and assessing the incremental value of a new (bio)marker. Heart. 2012;98(9):683-690.",
}

FIGURES = {
    "Figure 1": "figures/fig1_main_comparison.png",
    "Figure 2": "figures/fig_meta_analysis.png",
    "Figure 3": "figures/fig3_complexity_ladder.png",
    "Figure 4": "figures/fig_calibration.png",
    "Figure 5": "figures/fig_decision_analysis.png",
    "Figure 6": "figures/fig6_consequence_curves.png",
    "Figure 7": "figures/fig2_noise_degradation.png",
    "Figure 8": "figures/fig_external_learning_curves.png",
    "Figure 9": "figures/fig_simulation_study.png",
    "Figure S1": "figures/fig7_perfeature_drift.png",
    "Figure S2": "figures/fig8_shift_vs_degradation.png",
    "Figure S3": "figures/fig9_learning_curves.png",
    "Figure S4": "figures/fig_noise_types.png",
}


def resolve_citations(text: str) -> str:
    """Replace [@key, @key2] citations with numbered [n, m] / [n-m] forms."""
    order = []

    def repl(match):
        keys = [k.strip().lstrip("@") for k in match.group(1).split(",")]
        nums = []
        for key in keys:
            if key not in REFS:
                raise SystemExit(f"unknown citation key: {key}")
            if key not in order:
                order.append(key)
            nums.append(order.index(key) + 1)
        nums = sorted(set(nums))
        parts = []
        i = 0
        while i < len(nums):
            j = i
            while j + 1 < len(nums) and nums[j + 1] == nums[j] + 1:
                j += 1
            if j - i >= 2:
                parts.append(f"{nums[i]}-{nums[j]}")
            else:
                parts.extend(str(n) for n in nums[i:j + 1])
            i = j + 1
        return "[" + ", ".join(parts) + "]"

    text = re.sub(r"\[@([\w\s,@]+)\]", repl, text)

    refs_md = "\n".join(
        f"{i}. {REFS[key]}" for i, key in enumerate(order, 1)
    )
    text = text.replace("<!-- REFERENCES-GENERATED -->", refs_md)
    return text


def main() -> None:
    with open(SRC, encoding="utf-8") as f:
        text = f.read()
    resolved = resolve_citations(text)
    with open(RESOLVED, "w", encoding="utf-8") as f:
        f.write(resolved)

    subprocess.run(["pandoc", RESOLVED, "-o", OUT], check=True)
    doc = Document(OUT)

    # 1. Table rows: header repeats across pages, rows cannot split
    for table in doc.tables:
        for i, row in enumerate(table.rows):
            trPr = row._tr.get_or_add_trPr()
            if trPr.find(qn("w:cantSplit")) is None:
                trPr.append(OxmlElement("w:cantSplit"))
            if i == 0 and trPr.find(qn("w:tblHeader")) is None:
                header = OxmlElement("w:tblHeader")
                header.set(qn("w:val"), "true")
                trPr.insert(0, header)

    # 2. Table captions stay on the same page as their table
    for para in doc.paragraphs:
        if para.text.strip().startswith("Table"):
            pPr = para._p.get_or_add_pPr()
            if pPr.find(qn("w:keepNext")) is None:
                pPr.append(OxmlElement("w:keepNext"))

    # 3. Embed figure images directly below each figure caption
    paragraphs = doc.paragraphs
    for idx, para in enumerate(paragraphs):
        stripped = para.text.strip()
        for label, image_path in FIGURES.items():
            if stripped.startswith(label):
                if os.path.exists(image_path):
                    try:
                        if idx + 1 < len(paragraphs):
                            new_p = paragraphs[idx + 1].insert_paragraph_before()
                        else:
                            new_p = doc.add_paragraph()
                        new_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                        run = new_p.add_run()
                        run.add_picture(image_path, width=Inches(5.6))
                    except Exception as exc:
                        print(f"  (could not embed {image_path}: {exc})")
                else:
                    print(f"  (figure file missing: {image_path})")
                break

    doc.save(OUT)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
