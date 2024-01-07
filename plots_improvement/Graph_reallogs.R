

library(ggplot2)

setwd("C:/Users/ADMIN/Desktop/abc/results/KB_Modulation_results")
list_data= list.files( getwd())
cluster_data = list_data[which(grepl('cluster', list_data ,fixed = TRUE))]

data = 'helpdesk' 
# data = 'BPIC13_CP'
# data = 'BPIC13_I'
# data = 'Road_Traffic'
# data = 'sepsis_cases_1'
# data = 'BPIC12'

sel_data = cluster_data[which(grepl(data, cluster_data ,fixed = TRUE))]

for(w in 0:2){
  dat_fold1 = sel_data[which(grepl(paste0('fold',w), sel_data ,fixed = TRUE))]
  
  dat_fold1= dat_fold1[order(dat_fold1)]
  {
    fold1 = c()
    for(i in dat_fold1){
      dat = read.csv(i, T)
      fold1 = rbind(fold1, dat)
    }
    
    ave_fold1 = aggregate(fold1[, 'Damerau.Levenshtein'], by= list( fold1$Weight , fold1$Prefix.length), FUN= mean)
    
    prefix_start = aggregate(ave_fold1$x, by= list(ave_fold1$Group.2), FUN= function(x){x[1]})
    prefix_max = aggregate(ave_fold1$x, by= list(ave_fold1$Group.2), FUN= function(x){max(x)})
    prefix = cbind(prefix_start$x, prefix_max$x)
  }
  
  {
    ave_fold2 = aggregate(fold1[, 'Damerau.Levenshtein'], by= list( fold1$Weight , fold1$Prefix.length), FUN= length)
    see = unique(ave_fold2[,c(2,3)])
    names(see) = c("prefix", "count")
    ave_fold2$x = ave_fold2$x/sum(ave_fold2[which(ave_fold2$Group.1==0.0), 'x'])
    ave_fold1$x = ave_fold1$x*ave_fold2$x
    ave_fold1 = aggregate(ave_fold1[, c(3)], by= list( ave_fold1$Group.1), FUN= sum)
    ave_fold1 =  ave_fold1[which(ave_fold1$Group.1 < 1),]
    ave_fold1$x = round(ave_fold1$x,4)
  }

  nam <- paste0("ave_fold_", w)
  assign(nam, ave_fold1)
  
}

ave_fold = ave_fold1
ave_fold$x = (ave_fold_0$x + ave_fold_1$x + ave_fold_2$x)/3
ave_fold$x = round(ave_fold$x,3)

ggplot(data = ave_fold, aes(x=Group.1, y = x)) +
  geom_line() + 
  geom_point() +geom_text(aes(label = after_stat(y)), nudge_y= -0.02, nudge_x = 0.02, size = 4) +
  xlab("Weight")+ ylab("Similarity") +
  scale_x_continuous(breaks=  c( seq(0,0.95,0.05))  ) +
  theme(text = element_text(size=15),
        axis.text.x = element_text(angle=90)) 


