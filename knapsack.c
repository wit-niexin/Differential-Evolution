/*
 * ========================================
 * 差分进化算法求解0-1背包问题
 * ========================================
 * 
 * 问题描述：
 * 0-1背包问题是经典的组合优化问题。给定n个物品和一个容量为C的背包，
 * 每个物品i有重量w[i]和价值v[i]，要求选择若干物品放入背包，使得：
 * 1. 所选物品的总重量不超过背包容量C
 * 2. 所选物品的总价值最大
 * 3. 每个物品只能选择放入(1)或不放入(0)，不能部分放入
 * 
 * 数学模型：
 * 最大化：sum(v[i] * x[i])  其中 i = 0 到 n-1
 * 约束条件：sum(w[i] * x[i]) <= C
 * 决策变量：x[i] ∈ {0, 1}  (0表示不选，1表示选择)
 * 
 * 本例问题实例：
 * 背包容量：50
 * 物品数量：10
 * 物品信息：
 *   物品0: 重量=10, 价值=60
 *   物品1: 重量=20, 价值=100
 *   物品2: 重量=30, 价值=120
 *   ... (详见代码中的定义)
 * 
 * 算法适配说明：
 * 差分进化算法原本用于连续优化问题，为了求解0-1背包这种离散问题，
 * 我们采用以下策略：
 * 1. 个体编码：使用连续值[0,1]表示每个物品的"选择倾向"
 * 2. 解码方式：倾向值>=0.5表示选择该物品，<0.5表示不选择
 * 3. 约束处理：使用惩罚函数法，超重的解会被施加惩罚
 * 
 * 算法流程：
 * 1. 初始化种群（每个个体是一个[0,1]^n的连续向量）
 * 2. 解码为0-1方案并评估适应度
 * 3. 差分进化操作（变异、交叉、选择）
 * 4. 迭代进化直到满足终止条件
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <windows.h>

// ========================================
// 0-1背包问题参数
// ========================================
#define N_ITEMS 10         // 物品数量
#define CAPACITY 50        // 背包容量

// 物品的重量和价值（全局数组）
int weights[N_ITEMS] = {10, 20, 30, 5, 15, 25, 8, 12, 18, 22};
int values[N_ITEMS] = {60, 100, 120, 30, 80, 110, 40, 70, 90, 105};

// ========================================
// 差分进化算法参数设置
// ========================================
#define POP_SIZE 50        // 种群大小：种群中个体的数量
#define MAX_GEN 200        // 最大迭代次数：算法运行的代数
#define DIM N_ITEMS        // 问题维度：等于物品数量
#define F 0.8              // 缩放因子：控制差分向量的缩放程度
#define CR 0.9             // 交叉概率：控制参数交叉的概率
#define PENALTY 1000.0     // 惩罚系数：用于惩罚超重的解

// ========================================
// 早停机制参数设置
// ========================================
#define EARLY_STOP 1       // 早停开关：1=开启早停机制，0=关闭早停机制
#define NO_IMPROVE_LIMIT 50 // 无改进代数限制：连续多少代最优值无改进则提前终止

/*
 * ========================================
 * 解码函数：将连续编码转换为0-1方案
 * ========================================
 * 
 * 原理：
 * - 差分进化算法操作的是连续空间[0,1]^n的向量
 * - 需要将连续值转换为离散的0-1决策
 * - 转换规则：x[i] >= 0.5 表示选择物品i，否则不选择
 * 
 * 参数：
 *   continuous - 连续编码的个体（长度为N_ITEMS的double数组）
 *   binary - 输出的0-1方案（长度为N_ITEMS的int数组）
 */
void decode_solution(double *continuous, int *binary) {
    for (int i = 0; i < N_ITEMS; i++) {
        // 阈值解码：>= 0.5 选择，< 0.5 不选择
        binary[i] = (continuous[i] >= 0.5) ? 1 : 0;
    }
}

/*
 * ========================================
 * 目标函数（适应度函数）
 * ========================================
 * 
 * 功能：评估一个解的质量
 * 
 * 计算步骤：
 * 1. 将连续编码解码为0-1方案
 * 2. 计算总重量和总价值
 * 3. 如果超重，施加惩罚
 * 4. 返回适应度值（注意：我们要最大化价值，但DE默认最小化，所以返回负价值）
 * 
 * 惩罚机制：
 * - 如果总重量 <= 背包容量：适应度 = -总价值（负号是因为DE求最小值）
 * - 如果总重量 > 背包容量：适应度 = -总价值 + 惩罚系数 * 超重量
 * - 惩罚使得超重的解适应度变差，引导算法搜索可行解
 * 
 * 参数：
 *   x - 连续编码的个体
 * 返回值：
 *   适应度值（越小越好）
 */
double objective_function(double *x) {
    int binary[N_ITEMS];
    decode_solution(x, binary);
    
    int total_weight = 0;
    int total_value = 0;
    
    // 计算总重量和总价值
    for (int i = 0; i < N_ITEMS; i++) {
        if (binary[i] == 1) {
            total_weight += weights[i];
            total_value += values[i];
        }
    }
    
    // 计算适应度（带惩罚）
    double fitness;
    if (total_weight <= CAPACITY) {
        // 可行解：返回负价值（因为DE求最小值，我们要最大化价值）
        fitness = -total_value;
    } else {
        // 不可行解：施加惩罚
        int overweight = total_weight - CAPACITY;
        fitness = -total_value + PENALTY * overweight;
    }
    
    return fitness;
}

/*
 * ========================================
 * 随机数生成函数
 * ========================================
 * 生成[0,1]之间的均匀分布随机数
 */
double rand_double() {
    return (double)rand() / RAND_MAX;
}

/*
 * ========================================
 * 种群初始化函数
 * ========================================
 * 
 * 在[0,1]空间内随机生成初始种群
 * 每个个体的每个维度都是[0,1]之间的随机数
 * 
 * 参数：
 *   pop - 二维数组，存储整个种群，大小为POP_SIZE × N_ITEMS
 */
void initialize_population(double pop[][DIM]) {
    for (int i = 0; i < POP_SIZE; i++) {
        for (int j = 0; j < DIM; j++) {
            pop[i][j] = rand_double();  // [0,1]之间的随机数
        }
    }
}

/*
 * ========================================
 * 打印解的详细信息
 * ========================================
 * 
 * 功能：将最优解解码并打印详细信息
 * 
 * 参数：
 *   solution - 连续编码的解
 */
void print_solution(double *solution) {
    int binary[N_ITEMS];
    decode_solution(solution, binary);
    
    int total_weight = 0;
    int total_value = 0;
    
    printf("\n选择的物品：\n");
    printf("物品编号\t重量\t价值\t是否选择\n");
    printf("----------------------------------------\n");
    
    for (int i = 0; i < N_ITEMS; i++) {
        printf("物品%d\t\t%d\t%d\t%s\n", 
               i, weights[i], values[i], 
               binary[i] ? "是" : "否");
        if (binary[i] == 1) {
            total_weight += weights[i];
            total_value += values[i];
        }
    }
    
    printf("----------------------------------------\n");
    printf("总重量: %d / %d\n", total_weight, CAPACITY);
    printf("总价值: %d\n", total_value);
    
    if (total_weight > CAPACITY) {
        printf("警告: 超重 %d 单位！\n", total_weight - CAPACITY);
    }
}

/*
 * ========================================
 * 差分进化算法主函数
 * ========================================
 * 
 * 实现完整的差分进化算法流程，求解0-1背包问题
 */
void differential_evolution() {
    double population[POP_SIZE][DIM];
    double fitness[POP_SIZE];
    double trial[DIM];
    double best_solution[DIM];
    double best_fitness = INFINITY;
    
    // 早停机制相关变量
    int no_improve_count = 0;
    double prev_best_fitness = INFINITY;
    
    // ========================================
    // 步骤1：初始化种群
    // ========================================
    initialize_population(population);
    
    // ========================================
    // 步骤2：计算初始适应度
    // ========================================
    for (int i = 0; i < POP_SIZE; i++) {
        fitness[i] = objective_function(population[i]);
        if (fitness[i] < best_fitness) {
            best_fitness = fitness[i];
            for (int j = 0; j < DIM; j++) {
                best_solution[j] = population[i][j];
            }
        }
    }
    
    printf("初始最优价值: %d\n\n", (int)(-best_fitness));
    
    // ========================================
    // 步骤3：主循环 - 迭代进化
    // ========================================
    for (int gen = 0; gen < MAX_GEN; gen++) {
        prev_best_fitness = best_fitness;
        
        // 对每个个体进行进化操作
        for (int i = 0; i < POP_SIZE; i++) {
            // ========================================
            // 变异操作：DE/rand/1策略
            // ========================================
            int r1, r2, r3;
            do { r1 = rand() % POP_SIZE; } while (r1 == i);
            do { r2 = rand() % POP_SIZE; } while (r2 == i || r2 == r1);
            do { r3 = rand() % POP_SIZE; } while (r3 == i || r3 == r1 || r3 == r2);
            
            // ========================================
            // 交叉操作：二项式交叉
            // ========================================
            int j_rand = rand() % DIM;
            for (int j = 0; j < DIM; j++) {
                if (rand_double() < CR || j == j_rand) {
                    trial[j] = population[r1][j] + F * (population[r2][j] - population[r3][j]);
                    
                    // 边界处理：确保在[0,1]范围内
                    if (trial[j] < 0.0) trial[j] = 0.0;
                    if (trial[j] > 1.0) trial[j] = 1.0;
                } else {
                    trial[j] = population[i][j];
                }
            }
            
            // ========================================
            // 选择操作：贪婪选择
            // ========================================
            double trial_fitness = objective_function(trial);
            if (trial_fitness < fitness[i]) {
                for (int j = 0; j < DIM; j++) {
                    population[i][j] = trial[j];
                }
                fitness[i] = trial_fitness;
                
                // 更新全局最优
                if (trial_fitness < best_fitness) {
                    best_fitness = trial_fitness;
                    for (int j = 0; j < DIM; j++) {
                        best_solution[j] = trial[j];
                    }
                }
            }
        }
        
        // ========================================
        // 早停机制判断
        // ========================================
        if (EARLY_STOP) {
            if (fabs(best_fitness - prev_best_fitness) < 1e-10) {
                no_improve_count++;
            } else {
                no_improve_count = 0;
            }
            
            if (no_improve_count >= NO_IMPROVE_LIMIT) {
                printf("\n连续 %d 代无改进，提前终止！\n", NO_IMPROVE_LIMIT);
                printf("终止于第 %d 代\n", gen + 1);
                break;
            }
        }
        
        // 每20代输出一次结果
        if ((gen + 1) % 20 == 0) {
            printf("第 %d 代, 最优价值: %d", gen + 1, (int)(-best_fitness));
            if (EARLY_STOP) {
                printf(" (无改进计数: %d/%d)", no_improve_count, NO_IMPROVE_LIMIT);
            }
            printf("\n");
        }
    }
    
    // ========================================
    // 步骤4：输出最终结果
    // ========================================
    printf("\n========================================\n");
    printf("优化完成！\n");
    printf("========================================\n");
    print_solution(best_solution);
}

int main() {
    // 设置控制台输出为UTF-8编码，解决Windows下中文乱码问题
    SetConsoleOutputCP(65001);
    
    /*
     * 初始化随机数生成器
     * 使用当前时间作为种子，确保每次运行产生不同的随机数序列
     */
    srand((unsigned int)time(NULL));
    
    printf("========================================\n");
    printf("差分进化算法求解0-1背包问题\n");
    printf("========================================\n");
    printf("背包容量: %d\n", CAPACITY);
    printf("物品数量: %d\n", N_ITEMS);
    printf("种群大小: %d\n", POP_SIZE);
    printf("最大迭代次数: %d\n", MAX_GEN);
    if (EARLY_STOP) {
        printf("早停机制: 开启 (连续%d代无改进则终止)\n", NO_IMPROVE_LIMIT);
    } else {
        printf("早停机制: 关闭\n");
    }
    printf("========================================\n\n");
    
    // 打印物品信息
    printf("物品信息：\n");
    printf("物品编号\t重量\t价值\n");
    printf("----------------------------------------\n");
    for (int i = 0; i < N_ITEMS; i++) {
        printf("物品%d\t\t%d\t%d\n", i, weights[i], values[i]);
    }
    printf("========================================\n\n");
    
    differential_evolution();
    
    return 0;
}
