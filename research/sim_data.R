# specialty one
a <- 1
g <- 4
beta_slope <- 100
x1 <- round(runif(120, 0, 200))  
y1 <- beta_slope*((x1^a)/(x1^a + g^a)) 
plot(x1, y1)

a <- 4
g <- 300
beta_slope <- 80
x2 <- round(runif(120, 0, 1000))  
y2 <- beta_slope*((x2^a)/(x2^a + g^a)) 
plot(x2, y2)

a <- 0.5
g <- 50
beta_slope <- 60
x3 <- round(runif(120, 0, 500))  
y3 <- beta_slope*((x3^a)/(x3^a + g^a)) 
plot(x3, y3)

y <- round(y1 + y2 + y3)
plot(x3, y)

df_sim1 <- data.frame(Specialty = 'Neurologist', date = seq(120), rx_count = y, Email = x1, Phone = x2, Digital = x3)

# specialty two
a <- 0.1
g <- 0.5
beta_slope <- 65
x1 <- round(runif(120, 0, 200))  
y1 <- beta_slope*((x1^a)/(x1^a + g^a)) 
plot(x1, y1)

a <- 1
g <- 50
beta_slope <- 83
x2 <- round(runif(120, 0, 1000))  
y2 <- beta_slope*((x2^a)/(x2^a + g^a)) 
plot(x2, y2)

a <- 7
g <- 120
beta_slope <- 43
x3 <- round(runif(120, 0, 500))  
y3 <- beta_slope*((x3^a)/(x3^a + g^a)) 
plot(x3, y3)

y <- round(y1 + y2 + y3)
plot(x3, y)

df_sim2 <- data.frame(Specialty = 'Oncologist', date = seq(120), rx_count = y, Email = x1, Phone = x2, Digital = x3)

# specialty three
a <- 3
g <- 37
beta_slope <- 40
x1 <- round(runif(120, 0, 200))  
y1 <- beta_slope*((x1^a)/(x1^a + g^a)) 
plot(x1, y1)

a <- 2
g <- 200
beta_slope <- 40
x2 <- round(runif(120, 0, 1000))  
y2 <- beta_slope*((x2^a)/(x2^a + g^a)) 
plot(x2, y2)

a <- 1
g <- 15
beta_slope <- 180
x3 <- round(runif(120, 0, 500))  
y3 <- beta_slope*((x3^a)/(x3^a + g^a)) 
plot(x3, y3)

y <- round(y1 + y2 + y3)
plot(x3, y)

df_sim3 <- data.frame(Specialty = 'Hematologist', date = seq(120), rx_count = y, Email = x1, Phone = x2, Digital = x3)

df_final <- rbind(df_sim1, df_sim2, df_sim3)

write.csv(df_final, 'sim_data_r.csv')

plot(df_final$Email, df_final$rx_count)



