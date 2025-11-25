#=============================================================================
# 고객 이탈 예측 : 모델 전처리 및 시각화, 변수선택
# 수정: 25-11-22 23:15
#=============================================================================
#-----------------------------------------------------------------------
# 0. 데이터 불러오기
#-----------------------------------------------------------------------
churn <- read.csv("C:\\Users\\chs02\\OneDrive\\바탕 화면\\telecom_churn.csv")

dim(churn)     # 총 3333행 11열
colSums(is.na(churn)) # 결측치 확인 -> 결측치 없음

summary(churn) # 요약

# 0/1인 범주형 변수들 factor로 변환
churn$Churn <- factor(churn$Churn) # 고객 이탈 여부 (종속변수)
churn$ContractRenewal <- factor(churn$ContractRenewal)  # 계약 갱신 여부
churn$DataPlan        <- factor(churn$DataPlan)         # 데이터 플랜 여부

str(churn)     # 자료형 확인

#-----------------------------------------------------------------------
# 1. 시각화
#-----------------------------------------------------------------------
## 시각화1 - <고객 이탈 여부 막대그래프>
ggplot(churn, aes(x = Churn, fill = Churn)) +
  geom_bar(color = "black", width = 0.7) +
  geom_text(
    stat = "count",
    aes(label = ..count..),
    vjust = -0.3,
    size = 5,
    fontface = "bold"
  ) +
  scale_fill_manual(
    values = c("1" = "lightblue", "0" = "lightcoral")
  ) +
  labs(
    title = "고객 이탈(Churn) 비율",
    x = "Churn",
    y = "고객 수 (Count)"
  ) +
  scale_x_discrete(labels = c("0" = "유지(No)", "1" = "이탈(Yes)")) +
  theme_light()

#-----------------------------------------------------------------------
## 시각화2 - <계약 갱신 여부 및 플랜 보유 여부 vs 고객 이탈 비율 그래프>
library(patchwork)  

p1 <- ggplot(churn, aes(x = ContractRenewal, fill = Churn)) +
  geom_bar(position = "fill") +
  scale_fill_manual(
    values = c("1" = "lightblue", "0" = "lightcoral")
  ) +
  scale_x_discrete(labels = c("0" = "미갱신", "1" = "갱신")) +
  scale_y_continuous(labels = scales::percent_format()) +
  labs(
    title = "계약 갱신 여부에 따른 고객 이탈 비율",
    x = "ContractRenewal",
    y = "비율"
  ) +
  theme_light()

p2 <- ggplot(churn, aes(x = DataPlan, fill = Churn)) +
  geom_bar(position = "fill") +
  scale_fill_manual(
    values = c("1" = "lightblue", "0" = "lightcoral")
  ) +
  scale_x_discrete(labels = c("0" = "없음", "1" = "보유")) +
  scale_y_continuous(labels = scales::percent_format()) +
  labs(
    title = "데이터 플랜 보유 여부에 따른 고객 이탈 비율",
    x = "DataPlan",
    y = "비율"
  ) +
  theme_light()

p1 + p2

#-----------------------------------------------------------------------
## 시각화3 - <이탈 여부와 여러 숫자형 변수의 관계 비교 - 다중 박스플롯>
numeric_vars <- c("AccountWeeks", "DataUsage", "CustServCalls",
                  "DayMins", "DayCalls", "MonthlyCharge",
                  "OverageFee", "RoamMins")
  
# Churn을 No/Yes 순서로 정리
churn$Churn <- factor(churn$Churn, levels = c("0", "1"))

plot_list <- list()

for (v in numeric_vars) {
  
  p <- ggplot(churn, aes(x = Churn, y = .data[[v]], fill = Churn)) +
    geom_boxplot(outlier.size = 0.8) +
    scale_fill_manual(
      values = c("0" = "lightcoral", "1" = "lightblue")
    ) +
    labs(
      title = paste("Churn별", v, "분포"),
      x = "Churn",
      y = v
    ) +
    theme_light(base_size = 11) +
    theme(
      plot.title = element_text(size = 12, face = "bold"),
      legend.position = "none"
    )
  
  plot_list[[v]] <- p
}

# 한 화면에 정렬
wrap_plots(plot_list, ncol = 4)


#-----------------------------------------------------------------------
## 시각화4 - <상관 히트맵 시각화>
churn$Churn            <- as.numeric(as.character(churn$Churn))
churn$ContractRenewal  <- as.numeric(as.character(churn$ContractRenewal))
churn$DataPlan         <- as.numeric(as.character(churn$DataPlan))

# 모든 변수 (수치형 + 범주형 0/1)
numeric_vars <- churn[, sapply(churn, is.numeric)]

# 상관행렬 계산
corr_matrix <- cor(numeric_vars, use = "complete.obs")

# long 형태로 변환
library(reshape2)
corr_melt <- melt(corr_matrix)

# 상관 히트맵 시각화
library(ggplot2)
ggplot(corr_melt, aes(Var1, Var2, fill = value)) +
  geom_tile(color = "white") +
  geom_text(aes(label = round(value, 2)), size = 3, color = "black") +
  scale_fill_gradient2(low = "blue", mid = "white", high = "red",
                       midpoint = 0, limit = c(-1, 1), space = "Lab",
                       name = "Pearson\nCorrelation") +
  theme_light() +
  theme(
    axis.text.x = element_text(angle = 45, vjust = 1, hjust = 1),
    axis.title = element_blank(),
    panel.grid = element_blank()
  ) +
  ggtitle("Correlation Heatmap of Churn Dataset")
# DataUsage-DataPlan (0.95)로 매우 높은 상관관계를 지님
# ->	DataUsage를 남기고 DataPlan 제거


#-----------------------------------------------------------------------
# 2. 변수선택
#-----------------------------------------------------------------------
## [1단계] 강한 상관계수(0.95) 제거 -> (DataPlan)
select_vars <- setdiff(names(churn), "DataPlan")
select_vars

set.seed(123)
train_idx <- sample(1:nrow(churn), 0.8 * nrow(churn))
train <- churn[train_idx, select_vars]  # 2666개
test  <- churn[-train_idx, select_vars] # 667개

# 종속변수 factor 변환
train$Churn <- factor(train$Churn, levels = c(0,1))
test$Churn  <- factor(test$Churn,  levels = c(0,1))


## [2단계] LASSO로 데이터 기반 변수 선택
library(glmnet)
x <- model.matrix(Churn ~ ., train)[,-1]
y <- train$Churn

cv_lasso <- cv.glmnet(x, y, family="binomial", alpha=1)
lasso_coef <- coef(cv_lasso, s = "lambda.min"); lasso_coef
selected_vars <- rownames(lasso_coef)[which(lasso_coef != 0)]
selected_vars
# -> MonthlyCharge 삭제

select_vars2 <- setdiff(names(train), "MonthlyCharge")
select_vars2

train <- train[, select_vars2]

## [3단계] VIF 확인
library(car)
fit <- lm(as.numeric(train$Churn) ~ ., data = train)
vif(fit) 
# 다중공선성 없음

