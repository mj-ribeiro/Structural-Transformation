##################################################################################################################################
# 
#                                         Structural Transformation and Network Effects
#                                                    R&R Economic Modelling
# 
# #################################################################################################################################
# ### Marcos J Ribeiro
# ### Última atualização 14/05/2025
# #################################################################################################################################
# #### In this script I'll check the calibration results
#%%

{
  diretorio = dirname(rstudioapi::getSourceEditorContext()$path)
  setwd(diretorio)
  source('functions.r')
  library(readxl)
  library(tidyverse)
  library(ggrepel)

}


## dataset
df = readxl::read_xlsx('results_go_wgt.xlsx')

colnames(df)


df %>%
  filter(code != 'CHN') %>%
  ggplot(., aes(x =wgt_go_m,  y = wgt_go_d)) +
  geom_point() +
  geom_abline(intercept = 0, slope = 1, size = 0.5) 


df %>%
  filter(code != 'CHN') %>%
  ggplot(., aes(x =share_lab_m,  y = share_lab_d)) +
  geom_point() +
  geom_abline(intercept = 0, slope = 1, size = 0.5) +
  geom_text_repel(aes(label = code), size = 2.5, max.overlaps = 10) 


# correlation 
cor(df$share_lab_m, df$share_lab_d)


## desc stats tfp
df %>%
  group_by(sectors) %>%
  summarise(tfp_avg = mean(tfp),
            tfp_min = min(tfp),
            tfp_max = max(tfp),
            tfp_sd = sd(tfp)
            )



df %>%
  filter( code == 'AUS') %>%
  summarise(
    obj2 =  sum(((wgt_go_m-wgt_go_d) /wgt_go_d)^2)
  )








df %>%
  filter(code == 'CYP')












