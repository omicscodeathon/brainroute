# BrainRoute strict-validation results summary

## Data Accounting

| stage | count |
| --- | --- |
| starting_molecules | 9584 |
| invalid_smiles_removed | 13 |
| descriptor_calculation_failures_known | 0 |
| duplicate_molecules_found | 1593 |
| conflicting_label_molecules_removed | 69 |
| final_unique_molecules_available_for_modeling | 7888 |

## Duplicate Audit

| inchikey | n_entries | n_labels | labels | sources | canonical_smiles | is_duplicate |
| --- | --- | --- | --- | --- | --- | --- |
| AABLHGPVOULICI-BRJGLHKUSA-N | 2 | 1 | 1 | unspecified | CN1CC[C@]23c4c5ccc(O)c4O[C@H]2[C@@H](O)CC[C@@]3(O)[C@H]1C5 | True |
| AAOVKJBEBIDNHE-UHFFFAOYSA-N | 2 | 1 | 1 | unspecified | CN1C(=O)CN=C(c2ccccc2)c2cc(Cl)ccc21 | True |
| ACBLZFZDCOGNHD-UHFFFAOYSA-N | 2 | 1 | 1 | unspecified | CCCC(CCC)(CCC)C(N)=O | True |
| ACTOXUHEUCPTEW-OBURPCBNSA-N | 2 | 1 | 0 | unspecified | COC1C(O)CC(=O)OC(C)C/C=C/C=C/C(OC2CCC(N(C)C)C(C)O2)C(C)CC(CC=O)C1OC1OC(C)C(OC2CC(C)(O)C(O)C(C)O2)C(N(C)C)C1O | True |
| ADIMAYPTOBDMTL-UHFFFAOYSA-N | 2 | 1 | 1 | unspecified | O=C1Nc2ccc(Cl)cc2C(c2ccccc2)=NC1O | True |
| AEJOEPSMZCEYJN-FQEVSTJZSA-N | 2 | 1 | 1 | unspecified | CN(C(=O)Cc1ccc(Cl)c(Cl)c1)[C@@H](CN1CCCC1)c1ccccc1 | True |
| AEOOLRRTECSMIN-UHFFFAOYSA-N | 2 | 1 | 1 | unspecified | CNC1CCN2c3ccccc3CCc3cccc1c32 | True |
| AERLHOTUXIJQFV-RCPZPFRWSA-N | 2 | 1 | 1 | unspecified | O=C1[C@@H]2[C@@H]3C=C[C@@H]([C@H]4C=C[C@H]43)[C@@H]2C(=O)N1CCCCN1CCN(c2ncccn2)CC1 | True |
| AFBYHZACPPSJKD-UHFFFAOYSA-N | 2 | 1 | 1 | unspecified | CN(C)CCCN1c2ccccc2C=Cc2ccccc21 | True |
| AFZOCGNTFCGOEE-LROBGIAVSA-N | 2 | 1 | 1 | unspecified | CCC12CCN(CC3CC3)[C@@H](Cc3ccc(O)cc31)C2(C)C | True |
| AFZOCGNTFCGOEE-UHFFFAOYSA-N | 2 | 1 | 1 | unspecified | CCC12CCN(CC3CC3)C(Cc3ccc(O)cc31)C2(C)C | True |
| AGAHNABIDCTLHW-UHFFFAOYSA-N | 2 | 1 | 1 | unspecified | Cc1ccc(C2(O)CCN(CCCC(=O)c3ccc(F)cc3)CC2)cc1 | True |
| AGBTZJDOBMDLPR-UHFFFAOYSA-N | 2 | 1 | 1 | unspecified | CN1CCC2=C(CC1)c1cc(C#N)ccc1Sc1ccccc12 | True |
| AGOYDEPGAOXOCK-KCBOHYOISA-N | 2 | 1 | 0 | unspecified | CC[C@H]1OC(=O)[C@H](C)[C@@H](O[C@H]2C[C@@](C)(OC)[C@@H](O)[C@H](C)O2)[C@H](C)[C@@H](O[C@@H]2O[C@H](C)C[C@H](N(C)C)[C@H]2O)[C@](C)(OC)C[C@@H](C)C(=O)[C@H](C)[C@@H](O)[C@]1(C)O | True |
| AHCPKWJUALHOPH-UHFFFAOYSA-N | 2 | 1 | 1 | unspecified | Clc1cnn(CCCCN2CCN(c3ncccn3)CC2)c1 | True |
| AHDBQMJRRXVRDY-UHFFFAOYSA-N | 2 | 1 | 1 | unspecified | CC1(C)CN(CCN2CCN(c3cccc(Cl)c3)C2=O)C1 | True |
| AHKAOMZZTQULDS-UHFFFAOYSA-N | 2 | 1 | 1 | unspecified | O=C(OC1CN2CCC1CC2)c1ccccc1 | True |
| AHOUBRCZNHFOSL-YOEHRIQHSA-N | 2 | 1 | 1 | unspecified | Fc1ccc([C@@H]2CCNC[C@H]2COc2ccc3c(c2)OCO3)cc1 | True |
| AIJTTZAVMXIJGM-UHFFFAOYSA-N | 2 | 1 | 0 | unspecified | Cc1c(F)c(N2CCNC(C)C2)cc2c1c(=O)c(C(=O)O)cn2C1CC1 | True |
| AIUHRQHVWSUTGJ-UHFFFAOYSA-N | 2 | 1 | 1 | unspecified | CC(=O)OCCN1CCN(CCCN2c3ccccc3Sc3ccc(Cl)cc32)CC1 | True |

## Split Summary

| split | type | train_n | test_n |
| --- | --- | --- | --- |
| random80_seed42 | baseline_random | 6310 | 1578 |
| duplicate_aware_seed1 | duplicate_aware | 6310 | 1578 |
| duplicate_aware_seed2 | duplicate_aware | 6310 | 1578 |
| duplicate_aware_seed3 | duplicate_aware | 6310 | 1578 |
| duplicate_aware_seed4 | duplicate_aware | 6310 | 1578 |
| duplicate_aware_seed5 | duplicate_aware | 6310 | 1578 |
| scaffold_split_seed42 | scaffold_holdout | 6234 | 1654 |
| scaffold_cv_fold1 | primary_scaffold_cv | 6310 | 1578 |
| scaffold_cv_fold2 | primary_scaffold_cv | 6310 | 1578 |
| scaffold_cv_fold3 | primary_scaffold_cv | 6310 | 1578 |
| scaffold_cv_fold4 | primary_scaffold_cv | 6311 | 1577 |
| scaffold_cv_fold5 | primary_scaffold_cv | 6311 | 1577 |
| notebook_random_6040 | legacy_notebook_random_baseline | 5058 | 3491 |
| notebook_random_7030 | legacy_notebook_random_baseline | 5761 | 2674 |
| notebook_random_8020 | legacy_notebook_random_baseline | 6460 | 1814 |

## Near-Duplicate Similarity Summary

| split | mean_max_tanimoto | median_max_tanimoto | pct_gt_0.8 | pct_gt_0.85 | pct_gt_0.9 |
| --- | --- | --- | --- | --- | --- |
| duplicate_aware_seed1 | 0.7063151909365578 | 0.75 | 35.61470215462611 | 19.898605830164765 | 11.02661596958175 |
| duplicate_aware_seed2 | 0.6989237441266273 | 0.7398630136986302 | 32.953105196451205 | 19.328263624841572 | 11.660329531051964 |
| duplicate_aware_seed3 | 0.6971284387199285 | 0.7435897435897436 | 33.65019011406844 | 20.595690747782 | 12.103929024081117 |
| duplicate_aware_seed4 | 0.7056078934301739 | 0.75 | 34.09378960709759 | 19.455006337135615 | 11.913814955640053 |
| duplicate_aware_seed5 | 0.7054602673045235 | 0.75 | 33.840304182509506 | 19.835234474017746 | 11.913814955640053 |
| notebook_random_6040 | 0.750957778297546 | 0.7796610169491526 | 45.11601260383844 | 34.947006588370094 | 28.96018332855915 |
| notebook_random_7030 | 0.770172331075718 | 0.7970861486486487 | 48.952879581151834 | 38.51907255048616 | 32.16155572176515 |
| notebook_random_8020 | 0.7891622207303365 | 0.8110657827638961 | 52.315325248070565 | 42.28224917309812 | 35.501653803748624 |
| random80_seed42 | 0.7025385094679544 | 0.7486842105263158 | 35.551330798479086 | 20.215462610899873 | 12.420785804816225 |
| scaffold_cv_fold1 | 0.5032659126127231 | 0.5142857142857142 | 1.394169835234474 | 0.5069708491761723 | 0.1901140684410646 |
| scaffold_cv_fold2 | 0.4697740152603132 | 0.4464285714285714 | 0.9505703422053232 | 0.4435994930291508 | 0.3168567807351077 |
| scaffold_cv_fold3 | 0.4474447344190118 | 0.4285714285714285 | 1.2674271229404308 | 0.8238276299112801 | 0.1901140684410646 |
| scaffold_cv_fold4 | 0.4979994932794862 | 0.5 | 1.5852885225110969 | 1.14140773620799 | 0.3804692454026633 |
| scaffold_cv_fold5 | 0.5177390349126721 | 0.5168539325842697 | 1.5218769816106532 | 0.9511731135066582 | 0.2536461636017755 |
| scaffold_split_seed42 | 0.4442089072304249 | 0.4240384615384615 | 2.357920193470375 | 1.0882708585247884 | 0.181378476420798 |

## Model Performance Summary

| feature_view | model | balanced_accuracy_mean | balanced_accuracy_std | auprc_mean | auprc_std | mcc_mean | mcc_std | f1_mean | f1_std | validation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| embeddings | extra_trees | 0.6938572871625077 | 0.0127533302955488 | 0.8901760698546278 | 0.03344336355681899 | 0.3896443300769665 | 0.022840278368420865 | 0.8099547755868113 | 0.03650872019118155 | scaffold_cv_primary |
| embeddings | knn | 0.6506040072209099 | 0.028184791051122968 | 0.8212331529165926 | 0.04769972054211151 | 0.30284134256182316 | 0.05094815864128446 | 0.7847431260487227 | 0.04382195114838009 | scaffold_cv_primary |
| embeddings | lightgbm | 0.7153460049800954 | 0.024564760870563418 | 0.8895726648075495 | 0.03168223345728003 | 0.422318987675092 | 0.049598621854884666 | 0.808759821036136 | 0.0427426929027201 | scaffold_cv_primary |
| embeddings | logistic_regression | 0.7019337892558573 | 0.026778933732261104 | 0.8626591492495882 | 0.03372672220329731 | 0.38130452436479495 | 0.0486121621853746 | 0.7658980502076641 | 0.05560038074162602 | scaffold_cv_primary |
| embeddings | random_forest | 0.6982917875822061 | 0.023457805327865067 | 0.8920851941270062 | 0.03345105443070812 | 0.39341707165010753 | 0.04334706367523203 | 0.8045094166381439 | 0.04229568598056615 | scaffold_cv_primary |
| embeddings | xgboost | 0.7103291190284299 | 0.022412256985316195 | 0.8845649679629837 | 0.03338089105445988 | 0.40506811302198387 | 0.04484392536161837 | 0.7951715988538541 | 0.042733639819053634 | scaffold_cv_primary |
| morgan | extra_trees | 0.7784526191226362 | 0.02339312624270684 | 0.935897333563369 | 0.022514357298939358 | 0.5493183057920359 | 0.05369001251956495 | 0.846969123215459 | 0.04536543715954254 | scaffold_cv_primary |
| morgan | knn | 0.6568591909823398 | 0.02742783887861477 | 0.815029416493158 | 0.04344574599857911 | 0.37863988732050996 | 0.0739317771795604 | 0.8320616746245066 | 0.042175408933710995 | scaffold_cv_primary |
| morgan | lightgbm | 0.7653105534580068 | 0.0321944814830853 | 0.9254344382925016 | 0.025551816239798394 | 0.5187678746196732 | 0.05917016765465288 | 0.8398860811096214 | 0.03836756655515615 | scaffold_cv_primary |
| morgan | logistic_regression | 0.6901779741169054 | 0.020230018743581317 | 0.8589950745814049 | 0.034145582454976044 | 0.36926855605142206 | 0.037021355849057434 | 0.7918616928092621 | 0.0316417461612526 | scaffold_cv_primary |
| morgan | random_forest | 0.7749843658311184 | 0.026649043021402835 | 0.9353977533163367 | 0.02065964004705165 | 0.5361595683843582 | 0.05516234863550876 | 0.8389063387638863 | 0.04322546494755525 | scaffold_cv_primary |
| morgan | xgboost | 0.7669834745264884 | 0.02850905403637318 | 0.9255776673232831 | 0.025577429095333476 | 0.5097105707996717 | 0.05321602641456438 | 0.8239710328515711 | 0.039502094362920084 | scaffold_cv_primary |
| padel | extra_trees | 0.7669176552703273 | 0.032080130159580174 | 0.9279536224949514 | 0.024650594989569546 | 0.5125376332315881 | 0.05767749140844228 | 0.8274923537702467 | 0.04266026687328059 | scaffold_cv_primary |
| padel | knn | 0.699584884782217 | 0.012362444559302646 | 0.8680930697616299 | 0.03816288307019366 | 0.4184367234902135 | 0.029260164632299995 | 0.8285518516860163 | 0.03670754829987981 | scaffold_cv_primary |
| padel | lightgbm | 0.7570644553196058 | 0.030105644493897948 | 0.9244181607551486 | 0.02366640337656253 | 0.49709950406531667 | 0.05340937663075702 | 0.8262822441759162 | 0.04026097456278625 | scaffold_cv_primary |
| padel | logistic_regression | 0.7402645499546731 | 0.03013656153935887 | 0.8921434964824391 | 0.026610428062439177 | 0.4581791876293152 | 0.057048131573062495 | 0.802723617845151 | 0.05111642613225979 | scaffold_cv_primary |
| padel | random_forest | 0.7630118510789254 | 0.032537652556621395 | 0.9228947947593126 | 0.02614604090809712 | 0.5035782254941852 | 0.05973444372722213 | 0.8220093954760005 | 0.04366556913828095 | scaffold_cv_primary |
| padel | xgboost | 0.7577450608866856 | 0.03805737411319317 | 0.9222158678159916 | 0.02607070590820913 | 0.4912015995459832 | 0.06956214698271479 | 0.8155959635219221 | 0.04466817996495517 | scaffold_cv_primary |
| padel_morgan | extra_trees | 0.7761886621323981 | 0.034155098533268646 | 0.9372593934889007 | 0.02213434518255011 | 0.5324557960414845 | 0.06392406923637729 | 0.8349513550502303 | 0.044323276843060956 | scaffold_cv_primary |
| padel_morgan | knn | 0.7294081341806364 | 0.029917758632585896 | 0.8659489051987326 | 0.03784843995744317 | 0.47705368100059947 | 0.04714331371881797 | 0.8463719535016583 | 0.025989095014617054 | scaffold_cv_primary |

## External Validation

Not available.

## Statistical Comparison

| comparison | metric | feature_view | model | mean | std | count | ci95_low | ci95_high | p_value_wilcoxon |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| summary | balanced_accuracy | embeddings | extra_trees | 0.6938572871625077 | 0.0127533302955488 | 5.0 | 0.6826785002782088 | 0.7050360740468067 |  |
| summary | balanced_accuracy | embeddings | knn | 0.6506040072209099 | 0.0281847910511229 | 5.0 | 0.62589894860191 | 0.6753090658399099 |  |
| summary | balanced_accuracy | embeddings | lightgbm | 0.7153460049800954 | 0.0245647608705634 | 5.0 | 0.6938140427183136 | 0.7368779672418772 |  |
| summary | balanced_accuracy | embeddings | logistic_regression | 0.7019337892558573 | 0.0267789337322611 | 5.0 | 0.6784610189092605 | 0.725406559602454 |  |
| summary | balanced_accuracy | embeddings | random_forest | 0.6982917875822061 | 0.023457805327865 | 5.0 | 0.6777301146343094 | 0.7188534605301028 |  |
| summary | balanced_accuracy | embeddings | xgboost | 0.7103291190284299 | 0.0224122569853161 | 5.0 | 0.6906839096102722 | 0.7299743284465876 |  |
| summary | balanced_accuracy | morgan | extra_trees | 0.7784526191226362 | 0.0233931262427068 | 5.0 | 0.7579476398925448 | 0.7989575983527274 |  |
| summary | balanced_accuracy | morgan | knn | 0.6568591909823398 | 0.0274278388786147 | 5.0 | 0.6328176301966101 | 0.6809007517680695 |  |
| summary | balanced_accuracy | morgan | lightgbm | 0.7653105534580068 | 0.0321944814830853 | 5.0 | 0.7370908462121644 | 0.7935302607038492 |  |
| summary | balanced_accuracy | morgan | logistic_regression | 0.6901779741169054 | 0.0202300187435813 | 5.0 | 0.6724455808549823 | 0.7079103673788285 |  |
| summary | balanced_accuracy | morgan | random_forest | 0.7749843658311184 | 0.0266490430214028 | 5.0 | 0.7516254497124985 | 0.7983432819497381 |  |
| summary | balanced_accuracy | morgan | xgboost | 0.7669834745264884 | 0.0285090540363731 | 5.0 | 0.7419941868690667 | 0.7919727621839101 |  |
| summary | balanced_accuracy | padel | extra_trees | 0.7669176552703273 | 0.0320801301595801 | 5.0 | 0.7387981813788932 | 0.7950371291617614 |  |
| summary | balanced_accuracy | padel | knn | 0.699584884782217 | 0.0123624445593026 | 5.0 | 0.688748724352369 | 0.7104210452120648 |  |
| summary | balanced_accuracy | padel | lightgbm | 0.7570644553196058 | 0.0301056444938979 | 5.0 | 0.730675694422445 | 0.7834532162167666 |  |
| summary | balanced_accuracy | padel | logistic_regression | 0.7402645499546731 | 0.0301365615393588 | 5.0 | 0.7138486890723091 | 0.766680410837037 |  |
| summary | balanced_accuracy | padel | random_forest | 0.7630118510789254 | 0.0325376525566213 | 5.0 | 0.7344913411245343 | 0.7915323610333166 |  |
| summary | balanced_accuracy | padel | xgboost | 0.7577450608866856 | 0.0380573741131931 | 5.0 | 0.7243863016662873 | 0.7911038201070839 |  |
| summary | balanced_accuracy | padel_morgan | extra_trees | 0.7761886621323981 | 0.0341550985332686 | 5.0 | 0.74625039826975 | 0.8061269259950462 |  |
| summary | balanced_accuracy | padel_morgan | knn | 0.7294081341806364 | 0.0299177586325858 | 5.0 | 0.7031840625021742 | 0.7556322058590986 |  |
