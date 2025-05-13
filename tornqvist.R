##################################################################################################################################
# 
#                                         Structural Transformation and Network Effects
#                                                    R&R Economic Modelling
# 
# #################################################################################################################################
# ### Marcos J Ribeiro
# ### Última atualização 01/04/2025
# #################################################################################################################################
# #### In this script I calculated the increase rate of Tonrqvist index (see Marc's suggestions)


#%%

{
  diretorio = dirname(rstudioapi::getSourceEditorContext()$path)
  setwd(diretorio)
  source('functions.r')
  library(readxl)
  library(tidyverse)
  library(plm)
  library(WDI)
  library(fixest)
  library(modelsummary)
  
}


#%%
######################## GGDC DATASET ####################################

### Get datasets

# sectors classification
set_names =  readxl::read_xlsx("GGDC.xlsx", sheet = 'set_names')

# Get and prepare GGDC Dataset
data =  readxl::read_xlsx("GGDC.xlsx", sheet = 'dataset') %>%
  select(-`Summation of sector GDP`) %>%
  pivot_longer(cols = Agriculture:'Community, social and personal services',
               values_to = 'value',
               names_to = 'subsector') %>%
  left_join(., set_names, by = join_by('subsector' == 'subsector')) %>%
  pivot_wider(., names_from = 'variable', values_from = value)
  




#%%

## Get data from USA

data_usa = data %>%
  rename(code = country) %>%
  select(code, sector, year, VA, EMP) %>%
  group_by(code, sector, year) %>%
  reframe(VA = sum(VA),
          EMP = sum(EMP)
        ) %>%
  filter(
          code == 'USA'
          & year %in% seq(1950, 2020, 10)
        ) %>%
  group_by(year) %>%
  mutate(
    share_lab_d = EMP/sum(EMP),
    # VA_emp = (VA/EMP),
    # VA_emp = VA_emp/VA_emp[sector == 'Agriculture']
  ) %>%
  group_by(sector) %>%
  mutate(
      VA_emp = VA/EMP,
      VA_emp = VA_emp/VA_emp[year == 1950]
    ) %>%
  arrange(year, sector)



#%%
## SAVE US DATA

data_usa_tfp = data_usa %>%
  select(year, sector, VA_emp) %>%
  pivot_wider(names_from = year, values_from = VA_emp) %>%
  ungroup() %>%
  select(-sector)

writexl::write_xlsx(data_usa_tfp, 'data_usa_tfp.xlsx')


data_usa_labsh = data_usa %>%
  select(year, share_lab_d) %>%
  pivot_wider(names_from = year, values_from = share_lab_d) %>%
  ungroup() %>%
  select(-sector)

writexl::write_xlsx(data_usa_labsh, 'data_usa_labsh.xlsx')



print('oi')



####################################
#%%

### plot
data %>%
  filter(country == 'USA' 
          & !is.na(VA)
          & !is.na(VA_Q05/EMP)      
          & subsector %in% c('Agriculture', 
                              'Mining', 
                              'Government services',
                              'Transport, storage and communication') 
                            ) %>%
  ggplot(., aes(x = year, y = VA)) +
  geom_line() +
  scale_x_continuous(breaks = seq(1950, 2023, 5)) +
  facet_wrap('sector') +
  theme_minimal() +
  theme(axis.text.x = element_text(size=10, angle = 45))



























##### I won't use the code bellow
#%%

## Calculate Tornqvist Index (GGDC)
data = data %>%
  group_by(country, subsector) %>%
  mutate(
    A_l = VA_Q05/EMP,
    #lag_Al = dplyr::lag(A_l),
    gamma_l = log(A_l/dplyr::lag(A_l) )
  ) %>%
  group_by(country, sector) %>%
  mutate(
    #lag_VA = dplyr::lag(VA),
    omega = 0.5*( VA/sum(VA, na.rm = T) + dplyr::lag(VA)/sum(dplyr::lag(VA), na.rm = T) ),
  ) %>% 
  group_by(country, year, sector) %>%
  mutate(gamma = sum(omega*gamma_l, na.rm = T)) %>%
  ungroup()


  
#%%

# Only some checks
tt1 %>%
  filter(country == 'BRA' 
          #& year == 2010
          & !is.na(VA)
          & !is.na(VA_Q05/EMP)                  
          & subsector %in% c('Agriculture', 
                              'Mining', 
                              'Government services',
                              'Transport, storage and communication')
          & sector == 'Agriculture' 
                            ) %>%
  select(country, year, VA, lag_VA, A_l, lag_Al, gamma_l, omega, gamma) 



#%%
######################## WIOD DATASET ####################################

## Get sectors and subsectors names
set_names =  readxl::read_xlsx("sea.xlsx", sheet = 'sectors')

## GET WIOD DATASET
data2 = readxl::read_xlsx("sea.xlsx", sheet = 'DATA') %>%
          pivot_longer(!c(country, variable, description, code), 
                       names_to = "year", 
                       values_to = "value") %>%
          left_join(., set_names, by = 'description') %>%
          mutate(year = as.numeric(year)) %>%
          pivot_wider(., names_from = 'variable', values_from = value) %>%
          rename(subsector = description)


#%%
 
## Tornqvist (WIOD Dataset)

data2 = data2 %>%
  group_by(country, subsector) %>%
  mutate(
    A_l = VA_QI/EMPE,
    lag_Al = dplyr::lag(A_l),
    gamma_l = A_l/dplyr::lag(A_l) - 1 # there are negative values on VA
  ) %>%
  group_by(country, sector) %>%
  mutate(
    lag_VA = dplyr::lag(VA),
    omega = 0.5*( VA/sum(VA, na.rm = T) + dplyr::lag(VA)/sum(dplyr::lag(VA), na.rm = T) ),
  ) %>% 
  group_by(country, year, sector) %>%
  mutate(gamma = sum(omega*gamma_l, na.rm = T)) %>%
  ungroup()


#%%

## some plots

tt2 %>%
  filter(country == 'USA' 
          & !is.na(VA)
          & !is.na(VA_QI/EMPE)      
          & subsector %in% c('Forestry and logging', 
                              'Mining and quarrying', 
                              'Water transport',
                              'Education') 
                            ) %>%
  ggplot(., aes(x = year, y = 100*gamma)) +
  geom_line() +
  scale_x_continuous(breaks = seq(1950, 2023, 5)) +
  facet_wrap('sector') +
  theme_minimal() +
  theme(axis.text.x = element_text(size=10, angle = 45))






