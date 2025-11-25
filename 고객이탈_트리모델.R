#=============================================================================
# 고객 이탈 예측 : 의사결정나무(트리) 모델
# 수정: 25-11-25 00:04
#=============================================================================
library(caret)
library(ggplot2)

#-----------------------------------------------------------------------
# 0. 데이터 불러오기
#-----------------------------------------------------------------------
churn <- read.csv("C:\\Users\\chs02\\OneDrive\\바탕 화면\\telecom_churn.csv")

#-----------------------------------------------------------------------
# 1. 학습/검증 데이터 분할
#-----------------------------------------------------------------------
select_vars <- setdiff(names(churn), c("DataPlan", "MonthlyCharge"))
select_vars

set.seed(123)
train_idx <- sample(1:nrow(churn), 0.8 * nrow(churn))
train <- churn[train_idx, select_vars]
test  <- churn[-train_idx, select_vars]

# 종속변수 factor 변환
train$Churn <- factor(train$Churn, levels = c(0,1))
test$Churn  <- factor(test$Churn,  levels = c(0,1))

#-----------------------------------------------------------------------
# 2. 공통 교차검증 설정 (5-fold, 1회 반복)
#-----------------------------------------------------------------------
ctrl <- trainControl(method="cv", number=5)

#-----------------------------------------------------------------------
# 3. 의사결정나무
#-----------------------------------------------------------------------
library(rpart)
library(rpart.plot)

## (3-1) 완전 트리 생성 (cp = 0 → 최대한 깊게 분기)
set.seed(123)
tree.model <- rpart(
  Churn ~ .,
  data   = train,
  method = "class",
  cp     = 0,        # 가지치기 없이 전체 트리 생성
  xval   = 5         # 5-fold cross-validation
)

tree.model$cptable   # cp table 확인 (xerror, xstd 포함)

## (3-2) Test 데이터 예측 (가지치기 전)
tree.pred <- predict(tree.model, newdata = test, type = "class")
acc_tree  <- confusionMatrix(tree.pred, test$Churn)$overall[1]
acc_tree   # 가지치기전: 0.928

## (3-3) 최적 cp 선택
cpt <- as.data.frame(tree.model$cptable)

# (3-3-A) min(xerror) 기준 cp
min_xerror <- min(cpt[, "xerror"])
cp_min     <- cpt[which.min(cpt[, "xerror"]), "CP"]
cp_min # 0.005194805

# (3-3-B) 1-SE rule 기준 cp
# -> min(xerror) + xstd(min_xerror 위치)
threshold   <- min_xerror + cpt[which.min(cpt$xerror), "xstd"]
cp_1se      <- max(cpt[cpt$xerror <= threshold, "CP"])   # 가장 단순한 트리 선택
cp_1se  # 0.00909

## (3-4) 트리 가지치기
pruned.ct <- prune(tree.model, cp = cp_1se)

## (3-5) Test 성능 비교 (가지치기 후)
tree.pred2       <- predict(pruned.ct, newdata = test, type = "class")
cm_tree_pruned <- confusionMatrix(tree.pred2, test$Churn)
acc_tree_pruned  <- confusionMatrix(tree.pred2, test$Churn)$overall[1]

cm_tree_pruned    # 혼동행렬
acc_tree_pruned   # 0.9325337 (의사결정나무 최종 Test 정확도)

#-----------------------------------------------------------------------
# 4. 트리 결과 요약
#-----------------------------------------------------------------------
grid_cp <- expand.grid(cp = cp_1se) # CV 정확도, kappa를 얻기위해
set.seed(123)
fit_tree_cv <- train(
  Churn ~ .,
  data      = train,
  method    = "rpart",
  trControl = ctrl,
  tuneGrid  = grid_cp,
  metric    = "Accuracy"
)
fit_tree_cv

data_tree <- data.frame(
  # CV 성능 (resample 이용)
  CV_Accuracy   = mean(fit_tree_cv$resample$Accuracy),
  CV_Acc_SD     = sd(fit_tree_cv$resample$Accuracy),
  CV_Kappa      = mean(fit_tree_cv$resample$Kappa),
  CV_Kappa_SD   = sd(fit_tree_cv$resample$Kappa),
  
  # Test 성능 (confusionMatrix 이용)
  Test_Accuracy = cm_tree_pruned $overall["Accuracy"],
  Test_Kappa    = cm_tree_pruned $overall["Kappa"],
  Sensitivity   = cm_tree_pruned $byClass["Sensitivity"],
  Specificity   = cm_tree_pruned $byClass["Specificity"],
  Precision     = cm_tree_pruned $byClass["Precision"],
  Balanced_Acc  = cm_tree_pruned $byClass["Balanced Accuracy"],
  F1 = cm_tree_pruned $byClass["F1"]
)
rownames(data_tree) <- "tree"
data_tree

#-----------------------------------------------------------------------
# 5. 의사결정나무 트리 시각화
#-----------------------------------------------------------------------
## (5-1) <트리 구조 시각화>
# 완전 트리 (Pruning 전)
prp(
  tree.model,
  type = 2,              # 분기 조건 + 예측값
  extra = 2,             # 예측 클래스만 표시
  box.palette = "Blues",
  shadow.col  = "gray",
  cex = 0.7,
  branch = 0.7,
  compress = TRUE,
  space    = 0.4,
  fallen.leaves = TRUE,
  uniform = TRUE,
  round = 0.25,
  main = "Customer Churn Tree (Before Pruning)"
)

# 가지치기된 트리 (Pruned Tree)
prp(
  pruned.ct,
  type = 2,
  extra = 2,
  box.palette = "Blues",
  shadow.col  = "gray",
  cex = 0.8,
  branch = 0.7,
  compress = TRUE,
  space    = 0.4,
  fallen.leaves = TRUE,
  uniform = TRUE,
  round = 0.25,
  main = "Customer Churn Tree (After Pruning)"
)


## (5-2) <min(xerror) vs 1-SE Rule 시각화>
cpt$cp_index <- 1:nrow(cpt)   # x축 index

# min(xerror) 위치
min_idx <- which.min(cpt$xerror)
cp_min  <- cpt$CP[min_idx]

# 1-SE rule cp 위치
threshold <- cpt$xerror[min_idx] + cpt$xstd[min_idx]
cp_1se    <- max(cpt$CP[cpt$xerror <= threshold])
cp_1se_idx <- which(cpt$CP == cp_1se)

# 시각화
ggplot(cpt, aes(x = cp_index, y = xerror)) +
  geom_line(color = "darkblue", linewidth = 1) +
  geom_point(color = "darkblue", size = 2) +
  
  # min(xerror) 기준 표시
  geom_vline(xintercept = min_idx,
             color = "red",
             linetype = "dashed",
             linewidth = 1) +
  annotate("text",
           x = min_idx + 0.9,
           y = min(cpt$xerror),
           label = "min(xerror)",
           color = "red",
           vjust = -3,
           fontface = "bold") +
  
  # 1-SE rule 기준 표시
  geom_vline(xintercept = cp_1se_idx,
             color = "blue",
             linetype = "dashed",
             linewidth = 1) +
  annotate("text",
           x = cp_1se_idx + 0.9,
           y = min(cpt$xerror),
           label = "1-SE rule",
           color = "blue",
           vjust = -3,
           fontface = "bold") +
  
  labs(
    title = "min(xerror) vs 1-SE rule 비교",
    x = "CP Index",
    y = "Cross-Validation Error (xerror)"
  ) +
  theme_light() +
  theme(plot.title = element_text(face = "bold"))




