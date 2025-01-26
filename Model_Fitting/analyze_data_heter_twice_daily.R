# new analyze_data_heter

library(dplyr)
library(geepack)
library(glmnet)
library(readxl)

setwd("/Users/xuziping/Desktop/Research/ROADMAP2_Summaried/Fitting_Summaries")

pairs = read_excel("BMT_Roadmap_with_Peds_2023_04_21.xlsx")

file_location = "/Users/xuziping/Desktop/Research/ROADMAP2_Summaried/Fitting_Summaries/summary_stats_removing0"
impute_file_location = "/Users/xuziping/Desktop/Research/ROADMAP2_Summaried/Fitting_Summaries/imputed_data"
residual_file_location = "residuals_heter"
RData_file_location = "/Users/xuziping/Desktop/Research/ROADMAP2_Summaried/R_data"


##
missing_mood_thres = 0.85
missing_other_thres = 0.85
length_thres = 7*16
time_index = 1
missing_count = function(a){sum(is.na(a))/length(a)}

############################
## Aggregate data 
############################
aggregate = function(data0){
  n = dim(data0)[1]
  
  heartAM = apply(cbind(data0[2:n,"morningHEART"], data0[1:(n-1),"nightHEART"]), 1, mean, na.rm = TRUE)
  heartPM = data0[2:n,"dayHEART"]
  heartAM[is.nan(heartAM)]=NA
  heartPM[is.nan(heartPM)]=NA
  
  # mood is daily and AM and PM are the same
  moodAM = apply(cbind(data0[2:n,"morningMOOD"], data0[2:n,"dayMOOD"], data0[2:n,"nightMOOD"]), 1, mean, na.rm = TRUE)
  moodAM[is.nan(moodAM)]=NA
  moodPM = moodAM
  
  sleep_data = cbind(data0[2:n,"morningSLEEP"], data0[1:(n-1),"nightSLEEP"])
  sleepAM = apply(sleep_data, 1, sum, na.rm = TRUE) # * 2
  sleepAM[is.nan(sleepAM)]=NA
  sleepAM[sleepAM==0]=NA
  # there is generally no sleep data 12 before 9PM.
  sleepPM = sleepAM
  # sleepPM[is.nan(sleepPM)]=NA
  # sleepPM[sleepPM==0]=NA
  
  
  step_data = cbind(data0[2:n,"morningSTEPS"], data0[1:(n-1),"nightSTEPS"])
  stepAM = apply(step_data, 1, sum, na.rm = TRUE)
  stepAM[is.nan(stepAM)]=NA
  stepAM[stepAM==0]=NA
  stepPM = data0[2:n,"daySTEPS"]
  stepPM[is.nan(stepPM)]=NA
  stepPM[stepPM==0]=NA
  
  new_data = data.frame(cbind(heartAM, heartPM, moodAM, moodPM, sleepAM, sleepPM, stepAM, stepPM))
  return (new_data)
}
############################
## Compliance
############################
compliances = NULL

for (study_ID in c(201,203)){
  study_directory = sprintf("%s/%s/",file_location,study_ID)
  caregivers = list.files(path = study_directory)
  
  
  for (caregiver_number in (1:length(caregivers))){
    compliance = FALSE
    caregiver = substr(caregivers[caregiver_number],1,8)
    if (length(which(pairs[,3]==caregiver)) > 0){
      ## can find a match
      patient = toString(pairs[which(pairs[,3]==caregiver),2])
      filename_caregiver = sprintf("%s/%s/%s.Rda",file_location,study_ID, caregiver)
      load(filename_caregiver)
      data_caregiver0 = data_one_user
      data_caregiver = aggregate(data_caregiver0)
      filename_patient = sprintf("%s/%s/%s.Rda",file_location,202, patient)
      if (file.exists(filename_patient)){
        load(filename_patient)
        data_patient0 = data_one_user
        data_patient = aggregate(data_patient0)
        n = dim(data_patient)[1]
        if (n >= length_thres){
          
          missingness_caregiver = apply(data_caregiver[time_index:(time_index+length_thres-1),1:8], 2, missing_count)
          missingness_patient = apply(data_patient[time_index:(time_index+length_thres-1),1:8], 2, missing_count)
          
          if(
            (missingness_patient[3] < missing_mood_thres) &
            (missingness_caregiver[3] < missing_mood_thres) &
            (max(missingness_patient[c(1,2,5,7,8)]) < missing_other_thres)&
            (max(missingness_caregiver[c(1,2,5,7,8)]) < missing_other_thres)
          ){
            compliance = TRUE
          }
        }
        compliance_data = data.frame(compliance = compliance,
                                     study_ID = study_ID,
                                     caregiver = caregiver,
                                     patient = patient)
        compliances = rbind(compliances, compliance_data)
        
      }
    }
  }
}
#sum(compliances)
compliances

complianced_pairs = subset(compliances, compliance)
n_comp = dim(complianced_pairs)[1]
n_comp



############################
## Missing data imputation
############################
## missing data imputation


# first generate a raw dataset
n = dim(complianced_pairs)[1]
raw_data = NULL
for (pair_no in 1:n){
  print(pair_no)
  caregiver = complianced_pairs[pair_no,3]
  patient = complianced_pairs[pair_no,4]
  study_ID = complianced_pairs[pair_no,2]
  filename_caregiver = sprintf("%s/%s/%s.Rda",file_location,study_ID, caregiver)
  load(filename_caregiver)
  tmp_dat = aggregate(data_one_user)
  print(dim(tmp_dat))
  data_caregiver = tmp_dat[time_index:(time_index + length_thres),]
  ## save one more day to make sure that "next step" is well defined
  
  filename_patient = sprintf("%s/%s/%s.Rda",file_location,202, patient)
  load(filename_patient)
  tmp_dat = aggregate(data_one_user)
  print(dim(tmp_dat))
  data_patient = tmp_dat[time_index:(time_index + length_thres),]
  data_patient$idxNum = pair_no*2
  data_caregiver$idxNum = (pair_no-1)*2+1
  
  raw_data = rbind(raw_data, data_patient)
  raw_data = rbind(raw_data, data_caregiver)
}
new_filename = sprintf("%s/%s.Rda", RData_file_location, "raw_data")
save(raw_data, file = new_filename)

# impute missing data

library(mice)

n = dim(complianced_pairs)[1]
for (pair_no in 1:n){
  print(pair_no)
  caregiver = complianced_pairs[pair_no,3]
  patient = complianced_pairs[pair_no,4]
  study_ID = complianced_pairs[pair_no,2]
  filename_caregiver = sprintf("%s/%s/%s.Rda",file_location,study_ID, caregiver)
  load(filename_caregiver)
  data_caregiver = aggregate(data_one_user)[time_index:(time_index + length_thres),]
  ## save one more day to make sure that "next step" is well defined
  
  filename_patient = sprintf("%s/%s/%s.Rda",file_location,202, patient)
  load(filename_patient)
  data_patient = aggregate(data_one_user)[time_index:(time_index + length_thres),]
  
  data_caregiver_imputed = complete(mice(data_caregiver),1)
  data_caregiver_imputed$moodPM = data_caregiver_imputed$moodAM
  data_caregiver_imputed$sleepPM = data_caregiver_imputed$sleepAM
  
  data_patient_imputed = complete(mice(data_patient),1)
  data_patient_imputed$moodPM = data_patient_imputed$moodAM
  data_patient_imputed$sleepPM = data_patient_imputed$sleepAM
  
  new_filename = sprintf("%s/%s.Rda",impute_file_location, patient)
  
  save(data_caregiver_imputed, data_patient_imputed , file = new_filename)
}




############################
## Analyze data
############################

n = dim(complianced_pairs)[1]
full_data = NULL
full_data_weekly = NULL
for (pair_no in 1:n){
  #for (pair_no in 1){
  print(pair_no)
  caregiver = complianced_pairs[pair_no,3]
  patient = complianced_pairs[pair_no,4]
  study_ID = complianced_pairs[pair_no,2]
  
  filename_imputed = sprintf("%s/%s.Rda",impute_file_location, patient)
  load(filename_imputed)
  data_patient = data_patient_imputed
  data_caregiver = data_caregiver_imputed
  
  weekly_mood_patient0 = apply(matrix(data_patient[1:length_thres,3],nrow = 7), 2, mean, na.rm=TRUE)
  weekly_mood_caregiver0 = apply(matrix(data_caregiver[1:length_thres,3],nrow = 7), 2, mean, na.rm=TRUE)
  weekly_mood_patient = c(rep(NA,7), rep(weekly_mood_patient0[-length(weekly_mood_patient0)], each = 7))
  weekly_mood_caregiver = c(rep(NA,7), rep(weekly_mood_caregiver0[-length(weekly_mood_caregiver0)], each = 7))
  weekly_mood_patient_next = c(rep(weekly_mood_patient0, each = 7))
  weekly_mood_caregiver_next = c(rep(weekly_mood_caregiver0, each = 7))
  
  
  data_more = cbind(data_patient[1:length_thres, ], data_caregiver[1:length_thres, ], 
                    weekly_mood_patient, weekly_mood_caregiver, weekly_mood_patient_next, weekly_mood_caregiver_next,
                    weekly_mood_patient, weekly_mood_caregiver, weekly_mood_patient_next, weekly_mood_caregiver_next)
  colnames(data_more)[17:24] = c(
    "weeklyMoodAMPatient", "weeklyMoodAMCare", "weeklyMoodAMPatientNext", "weeklyMoodAMCareNext",
    "weeklyMoodPMPatient", "weeklyMoodPMCare", "weeklyMoodPMPatientNext", "weeklyMoodPMCareNext"
  )
  
  d = dim(data_patient)[2]
  colnames(data_more)[1:d] = paste0(colnames(data_more)[1:d], "Patient")
  colnames(data_more)[(d+1):(2*d)] = paste0(colnames(data_more)[(d+1):(2*d)], "Care")
  
  data_more[,"stepAMPatient"] = sqrt(data_more[,"stepAMPatient"])
  data_more[,"stepPMPatient"] = sqrt(data_more[,"stepPMPatient"])
  data_more[,"stepAMCare"] = sqrt(data_more[,"stepAMCare"])
  data_more[,"stepPMCare"] = sqrt(data_more[,"stepPMCare"])
  
  data_PM_next = rbind(data_more[-1,c("heartAMPatient", "moodAMPatient", "sleepAMPatient", "stepAMPatient",
                                          "heartAMCare", "moodAMCare", "sleepAMCare", "stepAMCare")], NA)
  data_AM_next = data_more[,c("heartPMPatient", "moodPMPatient", "sleepPMPatient", "stepPMPatient",
                              "heartPMCare", "moodPMCare", "sleepPMCare", "stepPMCare")]
  data_more = cbind(data_more, data_PM_next, data_AM_next)
  colnames(data_more)[25:40] = c(paste0(c("heartPMPatient", "moodPMPatient", "sleepPMPatient", "stepPMPatient",
                                          "heartPMCare", "moodPMCare", "sleepPMCare", "stepPMCare"), "Next"),
                                 paste0(c("heartAMPatient", "moodAMPatient", "sleepAMPatient", "stepAMPatient",
                                          "heartAMCare", "moodAMCare", "sleepAMCare", "stepAMCare"), "Next"))
  
  data_pair_no = rep(pair_no, length_thres)
  data_more = cbind(data_pair_no, data_more)
  colnames(data_more)[1] = "pairNo"
  data_which_group = rep(study_ID==201, length_thres)
  data_more = cbind(data_which_group, data_more)
  colnames(data_more)[1] = "whichGroup"
  
  
  
  data_final = data_more[1:length_thres,]
  full_data = rbind(full_data,data_final)
}




############################
## Standardize
############################

# generate adherence and relationship
full_data$ADHAMPatient = NA
full_data$ADHAMPatientNext = NA
full_data$ADHPMPatient = NA
full_data$ADHPMPatientNext = NA
full_data$Relationship = NA
full_data$RelationshipNext = NA


variables_to_save = c("stepAMPatient", "stepPMPatient")
step04 = apply(full_data[,variables_to_save], 2, function(x){quantile(x, probs = 0.4, na.rm = TRUE)})


  # save the average daily variables for the first week
  full_data$ADHAMPatient = as.numeric(full_data$stepAMPatient >= step04[1])
  full_data$ADHAMPatientNext = as.numeric(full_data$stepAMPatientNext >= step04[2])
  full_data$ADHPMPatient = as.numeric(full_data$stepPMPatient >= step04[2])
  full_data$ADHPMPatientNext = as.numeric(full_data$stepPMPatientNext >= step04[1])
  
  variables_to_save = c("weeklyMoodAMCare", "weeklyMoodAMPatient")
  mood05 = apply(data_final[(1:14)*7,variables_to_save], 2, function(x){quantile(x, probs = 0.5, na.rm = TRUE)})
  full_data$Relationship = as.numeric(full_data$weeklyMoodAMCare >= mood05[1]) * as.numeric(full_data$weeklyMoodAMPatient >= mood05[2])
  full_data$RelationshipNext = as.numeric(full_data$weeklyMoodAMCareNext >= mood05[1]) * as.numeric(full_data$weeklyMoodAMPatientNext >= mood05[2])

#########################################
## Add weighted average of adh and stress
#########################################


new_filename = sprintf("%s/%s.Rda",RData_file_location, "imputed_data")
save(full_data, file = new_filename)

vectors_to_stan = c("heartAMPatient",          "heartPMPatient",          "moodAMPatient",          
                    "moodPMPatient",           "sleepAMPatient",          "sleepPMPatient",          "stepAMPatient",           "stepPMPatient",          
                    "heartAMCare",             "heartPMCare",             "moodAMCare",              "moodPMCare",              "sleepAMCare",            
                    "sleepPMCare",             "stepAMCare",              "stepPMCare",              "weeklyMoodAMPatient",     "weeklyMoodAMCare",       
                    "weeklyMoodAMPatientNext", "weeklyMoodAMCareNext",    "weeklyMoodPMPatient",     "weeklyMoodPMCare",        "weeklyMoodPMPatientNext",
                    "weeklyMoodPMCareNext",    "heartPMPatientNext",      "moodPMPatientNext",       "sleepPMPatientNext",      "stepPMPatientNext",      
                    "heartPMCareNext",         "moodPMCareNext",          "sleepPMCareNext",         "stepPMCareNext",          "heartAMPatientNext",     
                    "moodAMPatientNext",       "sleepAMPatientNext",      "stepAMPatientNext",       "heartAMCareNext",         "moodAMCareNext",         
                    "sleepAMCareNext",         "stepAMCareNext")#,          "weeklyADH",               "weeklyStress")
full_data_stan = full_data %>% mutate_at(vectors_to_stan, ~(scale(.) %>% as.vector))

# generate weekly adherence and weekly stress
gamma = 6/7
for (pair_no in 1:n){
  full_data_stan$weeklyADH = stats::filter(full_data_stan$ADHAMPatient, gamma^seq(2, 14, 2), sides = 1) + stats::filter(full_data_stan$ADHPMPatient, gamma^seq(1, 14, 2), sides = 1)
  full_data_stan$weeklyStress = stats::filter(full_data_stan$sleepAMCare, gamma^seq(1, 7, 1), sides = 1) + stats::filter(full_data_stan$sleepPMCare, gamma^seq(1, 7, 1), sides = 1)
}

new_filename = sprintf("%s/%s.Rda",RData_file_location, "standardized_data")
save(full_data_stan, file = new_filename)

# save starting and generate binary
Sleep_Init = NA
for (pair_no in 1:n){
  print(pair_no)
  data_final = full_data_stan %>% dplyr::filter(pairNo == pair_no)
  #### save the original data!
  variables_to_save = c("sleepAMCare")
  data_original = data_final[8,variables_to_save]
  Sleep_Init = cbind(Sleep_Init, data_original)
}
Sleep_Init = Sleep_Init[1, 2:(n+1)]
filename = sprintf("%s/sleep_initial.csv",residual_file_location)
write.csv(Sleep_Init, filename)

save.image(file="gee.Rda")
