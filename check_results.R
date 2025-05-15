##################################################################################################################################
# 
#                                   Structural Transformation and Network Effects
#                                               R&R Economic Modelling
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
  library(countrycode)
  library(WDI)

}


### World Bank dataset
tt = WDIsearch('GDP', field = 'name')


wbk = WDI(
  indicator = c('NY.GDP.PCAP.PP.KD', 'SL.GDP.PCAP.EM.KD', 'NY.GDP.MKTP.KD'),
  start = 2014, 
  end = 2014,
  extra = TRUE) %>%
  tibble() %>%
  rename(gdp_pp = NY.GDP.PCAP.PP.KD,
         gdp_pw = SL.GDP.PCAP.EM.KD,
         gdp_data = NY.GDP.MKTP.KD) %>%
  select(iso3c, gdp_pp, gdp_pw, gdp_data, income) %>%
  rename(code = iso3c) 


## dataset
df = readxl::read_xlsx('results_go_wgt2.xlsx') %>%
      left_join(., wbk, by = 'code')




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
  ggplot(aes(x = tfp, y = log(gdp_pw) )) +
  geom_point() +
  geom_text_repel(aes(label = code), size = 2.5, max.overlaps = 10) +
  geom_smooth(method = 'lm', se = F) +
  facet_wrap(~sectors, nrow = 2, ncol = 2) +
  theme_light() +
  #my_theme +
  xlab('Calibrated TFP') +
  ylab('Log of GDP per person\n employment - World Bank')




## gdp model and data

df %>%
  mutate(
    GDP = GDP/GDP[code == 'USA'],
    gdp_data = gdp_data/gdp_data[code == 'USA']
  ) %>%
  filter(sectors == 'Agriculture') %>%
  ggplot(aes(x = gdp_data, y = GDP )) +
  geom_point() +
  geom_text_repel(aes(label = code), size = 2.5, max.overlaps = 10) +
  geom_smooth(method = 'lm', se = F) +
  ylab('GDP - Model')
  xlab('GDP - Data')
  
  
  



df %>%
  mutate(
    tfp_agg = share_lab_m*tfp,
    gdp_pw = gdp_pw/gdp_pw[code == 'USA']
  ) %>%
  filter(code != 'CYP') %>%
  group_by(code) %>%
  mutate(tfp_agg = sum(tfp_agg)) %>%
  filter(sectors == 'Agriculture') %>%
  ggplot(aes(x = gdp_pw, y = tfp_agg )) +
  geom_point() +
  geom_text_repel(aes(label = code), size = 2.5, max.overlaps = 10) +
  geom_smooth(method = 'lm', se = F) +
  ylab('Aggregate TFP') +
  xlab('GDP per worker (USA = 1)') +
  my_theme
  
  


