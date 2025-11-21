#=============================================================================
# 고객 이탈 예측 : 모델 전처리 및 시각화
# 수정: 25-11-20 22:17
#=============================================================================
#-----------------------------------------------------------------------
# 0. 데이터 불러오기
#-----------------------------------------------------------------------
churn <- read.csv("C:\\Users\\chs02\\OneDrive\\바탕 화면\\telecom_churn.csv")
churn

dim(churn)     # 총 3333행 11열
str(churn)     # 자료형 확인
summary(churn) # 요약
colSums(is.na(churn)) # 결측치 확인 -> 결측치 없음

#-----------------------------------------------------------------------
# 1. 시각화
#-----------------------------------------------------------------------
numeric_vars <- churn[, sapply(churn, is.numeric)]   # 숫자형만 추출

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
  theme_minimal(base_size = 12) +
  theme(
    axis.text.x = element_text(angle = 45, vjust = 1, hjust = 1),
    axis.title = element_blank(),
    panel.grid = element_blank()
  ) +
  ggtitle("Correlation Heatmap of Churn Dataset")

# DataUsage-DataPlan (0.95)로 매우 높은 상관관계를 지님
# ->	DataUsage를 남기고 DataPlan 제거

# 고객이탈 데이터 전체 산점도 행렬
library(GGally)
ggpairs(
  numeric_vars,
  title = "Scatterplot Matrix (Churn Dataset)",
  upper = list(continuous = wrap("cor", size = 3)),
  lower = list(continuous = wrap("points", alpha = 0.3, size = 1)),
  diag  = list(continuous = wrap("densityDiag"))
)


#-----------------------------------------------------------------------
# 2. 변수선택
#-----------------------------------------------------------------------
## [1단계] 강한 상관계수(0.95) 제거 -> (DataPlan)
select_vars <- setdiff(names(churn), "DataPlan")
select_vars

set.seed(123)
train_idx <- sample(1:nrow(churn), 0.8 * nrow(churn))
train <- churn[train_idx, select_vars]
test  <- churn[-train_idx, select_vars]

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

