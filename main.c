/*
 * ========================================
 * 差分进化算法 (Differential Evolution, DE)
 * ========================================
 * 
 * 算法简介：
 * 差分进化算法是一种基于群体的随机搜索优化算法，由Storn和Price于1995年提出。
 * 它通过模拟生物进化过程，利用种群中个体间的差异信息来指导搜索方向。
 * 
 * 核心思想：
 * 1. 维护一个候选解的种群
 * 2. 通过"变异"操作产生新的候选解（利用种群中个体的差分向量）
 * 3. 通过"交叉"操作增加解的多样性
 * 4. 通过"选择"操作保留更优的解
 * 5. 迭代进化，逐步逼近最优解
 * 
 * 算法优势：
 * - 参数少，易于使用
 * - 全局搜索能力强
 * - 对连续优化问题效果好
 * - 不需要目标函数的梯度信息
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <windows.h>

// ========================================
// 差分进化算法参数设置
// ========================================
#define POP_SIZE 50        // 种群大小：种群中个体的数量，越大搜索越全面但计算量越大
#define MAX_GEN 200        // 最大迭代次数：算法运行的代数，越多越可能找到最优解
#define DIM 2              // 问题维度：优化问题的变量个数（本例为2维）
#define F 0.8              // 缩放因子：控制差分向量的缩放程度，范围[0,2]，常用0.5-1.0
#define CR 0.9             // 交叉概率：控制参数交叉的概率，范围[0,1]，常用0.8-1.0
#define LOWER_BOUND -5.0   // 搜索空间下界：每个变量的最小取值
#define UPPER_BOUND 5.0    // 搜索空间上界：每个变量的最大取值

// ========================================
// 早停机制参数设置
// ========================================
#define EARLY_STOP 1       // 早停开关：1=开启早停机制，0=关闭早停机制
#define NO_IMPROVE_LIMIT 50 // 无改进代数限制：连续多少代最优值无改进则提前终止（仅当EARLY_STOP=1时生效）

/*
 * ========================================
 * 目标函数（适应度函数）
 * ========================================
 * 这是我们要优化的函数，差分进化算法的目标是找到使该函数值最小的x值
 * 
 * 本例使用Sphere函数：f(x) = x1^2 + x2^2
 * - 这是一个简单的凸函数，全局最优解在原点(0,0)
 * - 最小值为0
 * - 常用于测试优化算法的收敛性能
 * 
 * 参数：
 *   x - 长度为DIM的数组，表示一个候选解
 * 返回值：
 *   该候选解的目标函数值（适应度值）
 */
double objective_function(double *x) {
    double sum = 0.0;
    // 计算所有维度的平方和
    for (int i = 0; i < DIM; i++) {
        sum += x[i] * x[i];
    }
    return sum;
}

/*
 * ========================================
 * 随机数生成函数
 * ========================================
 * 生成[0,1]之间的均匀分布随机数
 * 用于种群初始化、个体选择和交叉操作
 */
double rand_double() {
    return (double)rand() / RAND_MAX;
}

/*
 * ========================================
 * 种群初始化函数
 * ========================================
 * 在搜索空间内随机生成初始种群
 * 
 * 原理：
 * - 种群是算法的搜索起点，需要在整个搜索空间内均匀分布
 * - 每个个体的每个维度都在[LOWER_BOUND, UPPER_BOUND]范围内随机生成
 * - 良好的初始化有助于算法更快找到最优解
 * 
 * 参数：
 *   pop - 二维数组，存储整个种群，大小为POP_SIZE × DIM
 */
void initialize_population(double pop[][DIM]) {
    for (int i = 0; i < POP_SIZE; i++) {           // 遍历每个个体
        for (int j = 0; j < DIM; j++) {            // 遍历每个维度
            // 在[LOWER_BOUND, UPPER_BOUND]范围内随机生成
            pop[i][j] = LOWER_BOUND + rand_double() * (UPPER_BOUND - LOWER_BOUND);
        }
    }
}

// 差分进化算法主函数
void differential_evolution() {
    double population[POP_SIZE][DIM];
    double fitness[POP_SIZE];
    double trial[DIM];
    double best_solution[DIM];
    double best_fitness = INFINITY;
    
    // ========================================
    // 早停机制相关变量
    // ========================================
    int no_improve_count = 0;      // 记录连续无改进的代数
    double prev_best_fitness = INFINITY;  // 记录上一代的最优值，用于判断是否有改进
    
    // 初始化种群
    initialize_population(population);
    
    // 计算初始适应度
    for (int i = 0; i < POP_SIZE; i++) {
        fitness[i] = objective_function(population[i]);
        if (fitness[i] < best_fitness) {
            best_fitness = fitness[i];
            for (int j = 0; j < DIM; j++) {
                best_solution[j] = population[i][j];
            }
        }
    }
    
    printf("初始最优值: %.6f\n\n", best_fitness);
    
    // 主循环
    for (int gen = 0; gen < MAX_GEN; gen++) {
        // 记录本代开始前的最优值
        prev_best_fitness = best_fitness;
        
        for (int i = 0; i < POP_SIZE; i++) {
            // 随机选择三个不同的个体
            int r1, r2, r3;
            do { r1 = rand() % POP_SIZE; } while (r1 == i);
            do { r2 = rand() % POP_SIZE; } while (r2 == i || r2 == r1);
            do { r3 = rand() % POP_SIZE; } while (r3 == i || r3 == r1 || r3 == r2);
            
            // 变异操作
            int j_rand = rand() % DIM;
            for (int j = 0; j < DIM; j++) {
                if (rand_double() < CR || j == j_rand) {
                    trial[j] = population[r1][j] + F * (population[r2][j] - population[r3][j]);
                    // 边界处理
                    if (trial[j] < LOWER_BOUND) trial[j] = LOWER_BOUND;
                    if (trial[j] > UPPER_BOUND) trial[j] = UPPER_BOUND;
                } else {
                    trial[j] = population[i][j];
                }
            }
            
            // 选择操作
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
            // 判断本代是否有改进（使用一个很小的阈值避免浮点数精度问题）
            if (fabs(best_fitness - prev_best_fitness) < 1e-10) {
                no_improve_count++;  // 无改进，计数器加1
            } else {
                no_improve_count = 0;  // 有改进，计数器清零
            }
            
            // 如果连续无改进代数达到限制，提前终止
            if (no_improve_count >= NO_IMPROVE_LIMIT) {
                printf("\n连续 %d 代无改进，提前终止！\n", NO_IMPROVE_LIMIT);
                printf("终止于第 %d 代\n", gen + 1);
                break;
            }
        }
        
        // 每20代输出一次结果
        if ((gen + 1) % 20 == 0) {
            printf("第 %d 代, 最优值: %.6f", gen + 1, best_fitness);
            if (EARLY_STOP) {
                printf(" (无改进计数: %d/%d)", no_improve_count, NO_IMPROVE_LIMIT);
            }
            printf("\n");
        }
    }
    
    // 输出最终结果
    printf("\n优化完成!\n");
    printf("最优解: (");
    for (int i = 0; i < DIM; i++) {
        printf("%.6f", best_solution[i]);
        if (i < DIM - 1) printf(", ");
    }
    printf(")\n");
    printf("最优值: %.6f\n", best_fitness);
}

int main() {
    // 设置控制台输出为UTF-8编码，解决Windows下中文乱码问题
    SetConsoleOutputCP(65001);
    
    /*
     * 初始化随机数生成器
     * 
     * 作用：
     * - srand()函数用于设置随机数种子，决定rand()函数生成的随机数序列
     * - time(NULL)获取当前时间戳（从1970年1月1日至今的秒数）
     * - 使用当前时间作为种子，确保每次运行程序时生成不同的随机数序列
     * 
     * - 如果不调用srand()，rand()每次运行程序都会生成相同的随机数序列
     * - 这会导致差分进化算法每次运行都得到相同的初始种群和相同的结果
     * - 使用时间戳作为种子，保证每次运行都有不同的随机性
     * 
     * 类型转换说明：
     * - time(NULL)返回time_t类型（在Windows上是long long int）
     * - srand()需要unsigned int类型参数
     */
    srand((unsigned int)time(NULL));
    
    printf("========================================\n");
    printf("差分进化算法求解优化问题\n");
    printf("========================================\n");
    printf("目标函数: f(x1, x2) = x1^2 + x2^2\n");
    printf("搜索空间: [%.1f, %.1f]\n", LOWER_BOUND, UPPER_BOUND);
    printf("种群大小: %d\n", POP_SIZE);
    printf("最大迭代次数: %d\n", MAX_GEN);
    if (EARLY_STOP) {
        printf("早停机制: 开启 (连续%d代无改进则终止)\n", NO_IMPROVE_LIMIT);
    } else {
        printf("早停机制: 关闭\n");
    }
    printf("========================================\n\n");
    
    differential_evolution();
    
    return 0;
}
