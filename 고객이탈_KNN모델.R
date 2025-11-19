#=============================================================================
# 고객 이탈 예측 : KNN모델
# 수정: 25-11-19 23:30
#=============================================================================
library(caret)
library(dplyr)
library(ggplot2)

#-----------------------------------------------------------------------
# 0. 데이터 불러오기
#-----------------------------------------------------------------------
churn <- read.csv("C:\\Users\\chs02\\OneDrive\\바탕 화면\\telecom_churn.csv")

#-----------------------------------------------------------------------
# 1. 학습/검증 데이터 분할
#-----------------------------------------------------------------------
set.seed(123)
train_idx <- sample(1:nrow(churn), 0.8 * nrow(churn))
train <- churn[train_idx, ]
test  <- churn[-train_idx, ]

# 종속변수 factor 변환
train$Churn <- factor(train$Churn, levels = c(0,1))
test$Churn  <- factor(test$Churn,  levels = c(0,1))

#-----------------------------------------------------------------------
# 2. 공통 교차검증 설정 (5-fold, 1회 반복)
#-----------------------------------------------------------------------
ctrl <- trainControl(method="cv", number=5)

#-----------------------------------------------------------------------
# 3. KNN (상관관계 0.8 이상 제거 + 표준화 + 최적 k 적용)
#-----------------------------------------------------------------------
## (3-1) 상관 높은 변수 제거 (타깃 Churn 제외)
cor_mat   <- cor(train[, -1])                     # 설명변수들만 상관계수 행렬
high_cor  <- caret::findCorrelation(cor_mat,
                                    cutoff = 0.8) # |r| > 0.8 변수 인덱스
selected_vars <- colnames(train)[-1][-high_cor]   # 선택된 설명변수 이름

# KNN용 학습/검증 데이터 (Churn + 선택된 변수만 사용)
train_knn <- train[, c("Churn", selected_vars)]
test_knn  <- test[,  c("Churn", selected_vars)]

## (3-2) KNN 튜닝 그리드 설정
grid_knn <- expand.grid(
  kmax     = seq(1, 15, by = 2),
  distance = c(1, 2),             # Manhattan, Euclidean
  kernel   = c("rectangular",     # 모두 동일 가중치
               "triangular",      # 가까운 이웃에 더 큰 가중치
               "gaussian")        # 가장 가까운 이웃에 큰 가중, 멀수록 거의 0
)

## (3-3) KNN 학습
set.seed(123)
fit_knn_all <- train(
  Churn ~ .,
  data       = train_knn,
  method     = "kknn",
  preProcess = c("center", "scale"), # 표준화
  tuneGrid   = grid_knn,
  trControl  = ctrl,
  metric     = "Accuracy"
)
fit_knn_all              # 전체 튜닝 결과
fit_knn_all$bestTune     # 최적 kmax, distance, kernel 조합

## (3-4) CV 성능 확인
# 1) bestTune 따로 저장
best <- fit_knn_all$bestTune

# 2) bestTune에 해당하는 CV 결과만 추출
best_cv <- subset(
  fit_knn_all$results,
  kmax == best$kmax &
    distance == best$distance &
    kernel == best$kernel
)
best_cv
# 정확도: 0.8987
# Kappa: 0.497

fit_knn_all$resample
knn_cv_acc <- mean(fit_knn_all$resample$Accuracy); knn_cv_acc # 0.8987
knn_cv_kap <- mean(fit_knn_all$resample$Kappa); knn_cv_kap # 0.497

## (3-5) 튜닝 결과 시각화
# caret 튜닝 결과 데이터프레임
fitknn_df <- fit_knn_all$results

# kmax를 숫자로 변환 (그래프 축에 자연스럽게 보이도록)
fitknn_df$kmax <- as.numeric(fitknn_df$kmax)

# distance를 보기 좋은 레이블로 변경
fitknn_df$distance <- factor(
  fitknn_df$distance,
  levels = c(1, 2),
  labels = c("Manhattan", "Euclidean")
)

# 최고 Accuracy 위치 및 값
best_idx <- which.max(fitknn_df$Accuracy)
best_k   <- fitknn_df$kmax[best_idx]
best_acc <- fitknn_df$Accuracy[best_idx]

# KNN 튜닝 결과 시각화
ggplot(fitknn_df,
       aes(x = kmax,
           y = Accuracy,
           color    = kernel,
           linetype = distance)) +
  geom_line(linewidth = 1) +
  geom_point(size = 2) +
  # 최적 k에 수직선 표시
  geom_vline(xintercept = best_k,
             color     = "red",
             linewidth = 0.2,
             linetype  = "dashed") +
  labs(
    title   = "KNN 튜닝 결과: kmax × distance × kernel",
    x       = "k (kmax)",
    y       = "Accuracy",
    color   = "Kernel",
    linetype = "Distance"
  ) +
  theme_light() +
  theme(
    plot.title  = element_text(face = "bold", size = 16),
    axis.title  = element_text(face = "bold")
  )

## (3-5) Test 데이터 최종 성능 평가
# 최적 튜닝(KNN 모형)으로 Test 데이터 예측
pred_knn <- predict(fit_knn_all, newdata = test_knn)

# 혼동행렬 및 정확도
cm_knn   <- caret::confusionMatrix(pred_knn, test_knn$Churn)
acc_knn  <- cm_knn$overall["Accuracy"]

cm_knn    # 혼동행렬
acc_knn   # 0.922039 (KNN 최종 Test 정확도)

#-----------------------------------------------------------------------
# 결과 요약
#-----------------------------------------------------------------------
data_knn <- data.frame(
    # CV 성능 (resample 이용)
    CV_Accuracy   = mean(fit_knn_all$resample$Accuracy),
    CV_Acc_SD     = sd(fit_knn_all$resample$Accuracy),
    CV_Kappa      = mean(fit_knn_all$resample$Kappa),
    CV_Kappa_SD   = sd(fit_knn_all$resample$Kappa),

    # Test 성능 (confusionMatrix 이용)
    Test_Accuracy = cm_knn$overall["Accuracy"],
    Test_Kappa    = cm_knn$overall["Kappa"],
    Sensitivity   = cm_knn$byClass["Sensitivity"],
    Specificity   = cm_knn$byClass["Specificity"],
    Precision     = cm_knn$byClass["Precision"],
    Balanced_Acc  = cm_knn$byClass["Balanced Accuracy"]
)
rownames(data_knn) <- "knn"
data_knn
