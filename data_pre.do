
*==================================Discovery data pre==================================

*-------------------construct raw_data-------------

*-F3NL adjust data for merge

import excel "F:\data\GNHS\data_pre\血样指标修正\F3NL血样指标整理.xls", sheet("F3NL") firstrow clear
save "D:\GNHS\data_pre\F3NL_blood_data",replace
import excel "F:\data\GNHS\data_pre\血样指标修正\F3NL报告201708-201904 整理- 2019-5-29.xlsx", sheet("F3NL问卷数据") firstrow clear
rename CODE1 id
merge 1:1 id using  "D:\GNHS\data_pre\F3NL_blood_data"
keep if _merge==3
drop _merge
gen SampleID_=substr(id,3,8)
egen wc_v4 = rmean(腰围1 腰围2)
egen hipc_v4 = rmean(臀围1 臀围2)
egen neck_v4 = rmean(颈围1 颈围2)
gen smoke_v4=是否抽烟
gen alc_v4=是否饮酒
gen tea_v4=是否喝茶
gen DM_med_v4=糖尿病服药
tab DM_med_v4,missing
rename 身高 height_v4
rename 体重 weight_v4
keep SampleID_-DM_med_v4 height_v4 weight_v4
save "D:\GNHS\data_pre\F3NL_adjust_data",replace 

*-mslc data drop

import excel "D:\GNHS\raw data\merge_mslc0306.xlsx", sheet("Sheet1") firstrow clear
drop GLU_v4 height_v4 weight_v4 TC_v4 LDL_v4 TG_v4 HDL_v4 GLU_v4 age age_v2 age_v3 age_v4 sequencingrun wc_v4 hipc_v4 neck_v4 smoke_v4 alc_v4 tea_v4 DM_med_v4
gen SampleID_=substr(SampleID,3,8)
duplicates drop SampleID_, force  //drop 34 obs
merge 1:1 SampleID_ using "D:\GNHS\data_pre\F3NL_adjust_data"
drop _merge
save "D:\GNHS\data_pre\mslc_data_add",replace

*-merge other necessary data

use "D:\GNHS\raw data\NL_data_analysis\GNHS_microdata20190527_otuRA.dta" ,clear
merge 1:1 SampleID using "D:\GNHS\raw data\NL_data_analysis\GNHS_phenotypedata20190530.dta" 
keep if _merge==3
drop _merge
gen SampleID_=substr(SampleID,3,8)
keep SampleID_ OTU0-OTU382874 RA_OTU0-RA_OTU382874 bristol_scale sequencingrun EA_solublefiber SBP_v4 DBP_v4 age-visit41 GLU_v4 TC_v4 LDL_v4 TG_v4 HDL_v4 GLU_v4
merge 1:1 SampleID_ using "D:\GNHS\data_pre\mslc_data_add",force
keep if _merge==3
drop _merge
save "D:\GNHS\data_pre\raw_data",replace

*----------------------------------------------------

*--------------------prepare data for T2D predict-------------

**-select val for T2D define

use "D:\GNHS\data_pre\raw_data",clear
keep SampleID DM_med DM_med_v2 med_v3 DM_med_v4 Glu GLU_v2 GLU_v3 GLU_v4 HBA1C_v2 hba1c_v3
save "D:\GNHS\data_pre\T2D_define",replace

**-merge outcome for raw data(GLU DM_outcome null replace with 999)

import excel "D:\GNHS\data_pre\T2D_outcome.xlsx", sheet("Sheet1") firstrow clear
merge 1:1 SampleID using "D:\GNHS\data_pre\raw_data",force
save "D:\GNHS\data_pre\raw_data_addT2D",replace

**-prepare data for T2D predict

use "D:\GNHS\data_pre\raw_data_addT2D",clear
qui destring GLU_v4-tea_v4,replace
keep SampleID GLU_v4 GLU_v3 DM_outcome3 DM_outcome4   observed_species-s__muciniphila  EA_solublefiber  edu3-fam_diabete energy MET marrige2   EA_whole_milk-EA_cheese veg fruit fish red_process_meat age_v4 SBP_v4 DBP_v4 LDL_v4 TG_v4 TC_v4 HDL_v4 sex_v4 height_v4 weight_v4 renal_dys_v4 ca_v4 wc_v4 hipc_v4 neck_v4 smoke_v4 alc_v4 tea_v4 age_v3 sex_v3 weight_v3 height_v3 BMI_v3 wc_v3 hipc_v3 neck_v3 SBP_v3 DBP_v3 smoke_v3 alc_v3 tea_v3 renal_dys_v3 ca_v3 TC_v3 LDL_v3 TG_v3 HDL_v3
gen dairy=EA_whole_milk+EA_skim_milk+EA_whole_milkpow+EA_skim_milkpow+EA_yogurt+EA_cheese
save "D:\GNHS\data_pre\data_for_predict",replace

*-split and reconstruct data for python

**-F2

use "D:\GNHS\data_pre\data_for_predict",clear //keep 1935
gen follow=real(substr(SampleID,2,1))
keep if follow==2 //keep 1263
keep SampleID  observed_species-s__muciniphila edu3 income4 fam_diabete-marrige2 EA_yogurt veg-red_process_meat dairy EA_solublefiber age_v3 sex_v3 weight_v3 height_v3 BMI_v3 wc_v3 hipc_v3 neck_v3 SBP_v3 DBP_v3  renal_dys_v3 ca_v3 TC_v3 LDL_v3 TG_v3 HDL_v3 DM_outcome3 GLU_v3 smoke_v3 alc_v3 tea_v3
renvars age_v3 sex_v3 weight_v3 height_v3 BMI_v3 wc_v3 hipc_v3 neck_v3 SBP_v3 DBP_v3  renal_dys_v3 ca_v3 TC_v3 LDL_v3 TG_v3 HDL_v3 DM_outcome3 GLU_v3 smoke_v3 alc_v3 tea_v3\age sex weight height BMI wc hipc neck SBP DBP  renal_dys ca TC LDL TG HDL DM_outcome GLU smoke alc tea
drop if ca==1 | renal_dys==1 //drop 34
drop if DM_outcome==999 //drop 45 keep 1184
tab DM_outcome,missing //157 case
drop ca renal_dys 
save "D:\GNHS\data_pre\F2_predict_data",replace

**-F3

use "D:\GNHS\data_pre\data_for_predict",clear //keep 1935
gen follow=real(substr(SampleID,2,1))
keep if follow==3 //keep 672
keep SampleID GLU_v4  DM_outcome4  observed_species-s__muciniphila EA_yogurt dairy EA_solublefiber  edu3 income4 fam_diabete-marrige2   veg-red_process_meat age_v4 SBP_v4 DBP_v4 LDL_v4 TG_v4 TC_v4 HDL_v4 sex_v4 height_v4 weight_v4 renal_dys_v4 ca_v4 wc_v4 hipc_v4 neck_v4 smoke_v4 alc_v4 tea_v4
renvars age_v4 SBP_v4 DBP_v4 LDL_v4 TG_v4 TC_v4 HDL_v4 sex_v4 height_v4 weight_v4 renal_dys_v4 ca_v4 wc_v4 hipc_v4 neck_v4  DM_outcome4 GLU_v4 smoke alc tea\age SBP DBP LDL TG TC HDL sex height weight renal_dys ca wc hipc neck   DM_outcome GLU smoke alc tea
drop if ca=="1" | renal_dys=="1" //drop 21
drop if DM_outcome==999 //drop 3 keep 648
tab DM_outcome,missing //113 case
drop ca renal_dys
save "D:\GNHS\data_pre\F3_predict_data",replace

**-predict data de

import excel "D:\GNHS\data_pre\data_for_predict.xlsx", sheet("Sheet1") firstrow clear //1832 obs
tab DM_outcome,missing //270 case
tab tea,missing
tab income4,missing
save "D:\GNHS\data_pre\data_input_predict.dta",replace

**-------------------------construct MRS---------------------------

*-merge the shap value and raw data

import excel "D:\GNHS\data_pre\data_select_shap_values.xlsx", sheet("Sheet1") firstrow clear
renvars fastingglucose-wc, postfix(_shap)  
drop DM_outcome
merge 1:1 SampleID using "D:\GNHS\data_pre\data_input_predict.dta"
drop _merge
save "D:\GNHS\data_pre\predict_val_add_shapvalue.dta",replace

*-construct MRS

use "D:\GNHS\data_pre\predict_val_add_shapvalue.dta",clear
global vars "c__alphaproteobacteria_shap c__deltaproteobacteria_shap f__comamonadaceae_shap-f__mogibacteriaceae_shap g__butyrivibrio_shap-g__roseburia_shap o__lactobacillales_shap-unweighted_nmds6_shap"
foreach v of varlist $vars{
            gen S_`v' = .
			replace S_`v'=0 if `v'<=0
			replace S_`v'=1 if `v'>0
         }
egen micro_score=rowtotal(S_c__alphaproteobacteria_shap-S_s__dispar_shap)

label var micro_score "Unweighted microbiota risk score"

save "D:\GNHS\data_pre\data_cross_analysis.dta",replace


*-----------------------prepare data for baslin-MRS analysis----------

use "D:\GNHS\data_pre\data_cross_analysis.dta",clear
keep SampleID micro_score q_micro_score w_micro_score q_w_micro_score neck DM_outcome dairy c__alphaproteobacteria_shap-wc_shap unweighted_nmds6 f__lactobacillaceae observed_species
merge 1:1 SampleID using "D:\GNHS\data_pre\raw_data",force
keep if _merge==3
keep SampleID micro_score q_micro_score w_micro_score q_w_micro_score DM_outcome  sequencingdepth age sex BMI weight height wc hipc neck  income4 edu3 marrige2 tea alc smoke red_process_meat  veg fruit fish EA_yogurt dairy energy MET c__alphaproteobacteria_shap-wc_shap unweighted_nmds6 f__lactobacillaceae observed_species
qui destring sex-sequencingdepth,replace

egen miss=rmiss(micro_score age-sequencingdepth neck)
drop if miss!=0 //drop 20 obs
global vars " veg fruit fish EA_yogurt red_process_meat "
foreach v of varlist $vars{
            xtile q_`v' = `v',nq(4)
         }
gen diet_score=q_veg+q_fruit+q_fish+EA_yogurt-q_red_process_meat
xtile q_diet_score=diet_score,nq(4)
xtile q_MET_score=MET,nq(4)
gen cat_BMI=.
replace cat_BMI=1 if BMI>=25
replace cat_BMI=0 if BMI<25
gen life_score=diet_score+q_MET_score
xtile q_life_score=life_score,nq(4)
save "D:\GNHS\data_pre\data_baln_MRS_analysis.dta",replace


*----------------------prepare data for MRS-future_glu/T2D analysis

use "D:\GNHS\data_pre\raw_data",clear
keep SampleID GLU_v4 DM_med_v4 wc_v4 neck_v4 hipc_v4 weight_v4 height_v4 
destring GLU_v4 DM_med_v4 wc_v4 neck_v4 hipc_v4 weight_v4 height_v4  ,replace
gen BMI_v4=weight_v4/(height_v4/100)^2
su BMI_v4,de
merge 1:1 SampleID using "D:\GNHS\data_pre\data_cross_analysis.dta"
keep if _merge==3
gen follow=real(substr(SampleID,2,1))
keep if follow==2 
keep if GLU_v4!=. //keep 188
su BMI_v4,de
//drop if DM_med_v4==1 | DM_med_v4==2 | DM_med_v4==3
gen DM_outcome_=0
replace DM_outcome_=1 if GLU_v4>=7 | DM_med_v4==1 | DM_med_v4==2 | DM_med_v4==3
tab DM_outcome_ DM_outcome //23 case,20ins
order DM_outcome_ DM_outcome q_micro_score
save "D:\GNHS\data_pre\MRS_glu_T2D_analysis.dta",replace
