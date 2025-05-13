##################################################################################################################################
# 
#                                         Structural Transformation and Network Effects
#                                                    R&R Economic Modelling
# 
# #################################################################################################################################
# ### Marcos J Ribeiro
# ### Última atualização 01/04/2025
# #################################################################################################################################
# #### In this script I make some usefull functions


#%%
{
  
  diretorio = dirname(rstudioapi::getSourceEditorContext()$path)
  setwd(diretorio)
  library(ggplot2)
}

#%%

### MY CONFIGURATION OF ggplot
my_theme = theme_bw() +
  theme(
    legend.position="bottom", 
    legend.box = 'horizontal',
    #panel.background = element_rect(fill = "linen"),
    #legend.background = element_blank(),
    plot.title = element_text(hjust = 0.5, size = 24, face = 'bold'),
    legend.text = element_text(size = 24),
    legend.title = element_blank(),
    strip.text.x = element_blank(),
    strip.background = element_rect(colour="white", fill="white"),
    #legend.position=c(0.25,0.95),
    axis.text.x = element_text(size=22, vjust = 0.5),
    axis.text.y = element_text(size=22),
    axis.title.x = element_text(colour = 'black', size=24),
    axis.title.y = element_text(colour = 'black', size=24) ) 


