
library(ggplot2)

setwd("C:/Users/ADMIN/Desktop/볼차노/Desktop/results/output")
list_data= list.files( getwd())


data = 'Synthetic' 
sel_data = list_data[which(grepl(data, list_data ,fixed = TRUE))]
sel_data

for(w in 1:3){
  fold1 = read.csv(sel_data[w], T)
  
  ave_fold1 = aggregate(fold1[, "Damerau.Levenshtein"], by= list( fold1$Weight , fold1$Prefix.length), FUN= mean)
  
  ave_fold1$Group.2 = as.character(ave_fold1$Group.2)
  names(ave_fold1)[2] = "prefix"
  
  
  fold1_one = fold1[which(fold1$Weight == 0 ), ]
  df_test_variant = data.frame(v_id = names(table(as.factor(fold1_one$Variant.ID))), count = as.vector(table(as.factor(fold1_one$Variant.ID))))
  see = df_test_variant[order(df_test_variant$count, decreasing =  TRUE),]
  row.names(see) <- NULL
  
  ave_fold1 = aggregate(fold1[, "Damerau.Levenshtein"], by= list( fold1$Weight , fold1$Prefix.length), FUN= mean)
  
  prefix_start = aggregate(ave_fold1$x, by= list(ave_fold1$Group.2), FUN= function(x){x[1]})
  prefix_max = aggregate(ave_fold1$x, by= list(ave_fold1$Group.2), FUN= function(x){max(x)})
  prefix = cbind(prefix_start$x, prefix_max$x)
  prefix
  
  ave_fold2 = aggregate(fold1[, "Damerau.Levenshtein"], by= list( fold1$Weight , fold1$Prefix.length), FUN= length)
  see = unique(ave_fold2[,c(2,3)])
  names(see) = c("prefix", "count")
  see
  ave_fold2$x = ave_fold2$x/sum(ave_fold2[which(ave_fold2$Group.1==0.0), 'x'])
  ave_fold1$x = ave_fold1$x*ave_fold2$x
  
  ave_fold1 = aggregate(ave_fold1[, c(3)], by= list( ave_fold1$Group.1), FUN= sum)
  ave_fold1 =  ave_fold1[which(ave_fold1$Group.1 < 1),]

  nam <- paste0("ave_fold_", w)
  assign(nam, ave_fold1)
  
}


ave_fold = ave_fold1
ave_fold$x = (ave_fold_1$x + ave_fold_2$x + ave_fold_3$x)/3
ave_fold$x = round(ave_fold$x,3)

ggplot(data = ave_fold, aes(x=Group.1, y = x)) +
  geom_line() + 
  geom_point() +geom_text(aes(label = after_stat(y)), nudge_y= -0.03, nudge_x = 0.02, size = 4) +
  # ylim(0.2, 1) +
  # ggtitle("fitness_token_based_replay") +
  xlab("Weight")+ ylab("Similarity") +
  ylim(0.2, 1) + scale_x_continuous(breaks=  c( seq(0,0.95,0.05))  ) +
  theme(text = element_text(size=15),
        axis.text.x = element_text(angle=90)) 


