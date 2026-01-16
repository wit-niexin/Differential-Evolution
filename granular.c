/*
 * ========================================
 * 粒度差分进化算法 (Granular Differential Evolution, Granular-DE)
 * ========================================
 * 
 * 核心思想：
 * 传统差分进化算法将决策变量视为D个独立的实数，在变异和交叉时逐维操作。
 * 这种方式忽略了变量之间可能存在的语义关联或结构依赖。
 * 
 * 粒度计算的引入：
 * 粒度差分进化将决策空间划分成若干"粒"（grain/block），每个粒包含若干相关的决策变量。
 * 在进化操作时，以"粒"为最小单位进行变异和交叉，而不是单个变量。
 * 
 * 优势：
 * 1. 保护优良片段：相关变量作为整体进化，减少破坏已形成的良好组合
 * 2. 引入高层知识：粒的划分可以体现问题的结构特征
 * 3. 粗-细两层搜索：粒级别是粗粒度搜索，粒内是细粒度优化
 * 4. 降低搜索复杂度：减少无效的变量组合尝试
 * 
 * 算法流程：
 * 1. 将D维决策空间划分成K个粒，每个粒包含若干维度
 * 2. 初始化种群
 * 3. 进化操作：
 *    a. 变异：按粒为单位计算差分向量
 *    b. 交叉：以粒为单位决定是否交叉（而非逐维）
 *    c. 选择：保留更优个体
 * 4. 迭代直到收敛
 * 
 * 本例应用：
 * 求解10维Schwefel函数的最小值
 * - 将10个维度划分成若干粒（如：每2维一个粒，共5个粒）
 * - 使用粒度差分进化算法搜索最优解
 * 
 * ========================================
 * Schwefel 2.26 函数介绍
 * ========================================
 * 
 * 函数定义：
 * f(x) = 418.9829 * D - Σ(x[i] * sin(sqrt(|x[i]|)))
 * 其中 i = 0 到 D-1
 * 
 * 特点：
 * - 高度多峰函数，有大量局部最优解
 * - 全局最优解在 x[i] = 420.9687 (所有维度)
 * - 全局最优值约为 0
 * - 搜索空间：[-500, 500]^D
 * - 难点：全局最优解远离原点，容易陷入局部最优
 * 
 * 为什么选择Schwefel 2.26函数：
 * - 经典的测试函数，广泛用于评估优化算法性能
 * - 多峰特性能够测试算法的全局搜索能力
 * - 适合展示粒度计算的优势（变量间存在相似的振荡模式）
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <windows.h>

// ========================================
// 问题参数设置
// ========================================
#define DIM 10             // 问题维度：Schwefel函数的维数
#define LOWER_BOUND -500.0 // 搜索空间下界
#define UPPER_BOUND 500.0  // 搜索空间上界

// ========================================
// 粒度划分参数
// ========================================
#define GRAIN_SIZE 2       // 粒大小：每个粒包含的维度数（10维分成5个粒，每粒2维）
#define N_GRAINS (DIM / GRAIN_SIZE)  // 粒的数量

/*
 * 粒划分说明：
 * 本例采用简单的连续划分策略：
 * - 粒0: 维度[0, 1]
 * - 粒1: 维度[2, 3]
 * - 粒2: 维度[4, 5]
 * - 粒3: 维度[6, 7]
 * - 粒4: 维度[8, 9]
 * 
 * 其他划分策略：
 * - 随机划分：随机分组
 * - 自适应划分：根据变量相关性动态调整
 * - 层次划分：多层次粒结构
 */

// ========================================
// 差分进化算法参数设置
// ========================================
/*
 * Schwefel函数优化难点及参数调整说明：
 * 
 * 难点：
 * 1. 全局最优解在420.9687附近，远离搜索空间中心(0)
 * 2. 函数表面有大量局部最优，容易陷入
 * 3. 需要较大的种群和足够的迭代次数
 * 
 * 参数调整策略：
 * - 增大种群：提高全局搜索能力
 * - 增加迭代次数：给算法更多时间收敛
 * - 调整F值：使用较大的F增强探索能力
 * - 适当降低CR：在粒度DE中，过高的CR可能导致过度探索
 */
#define POP_SIZE 1000      // 种群大小（增大以提高全局搜索能力）
#define MAX_GEN 2000       // 最大迭代次数（Schwefel函数需要更多迭代）
#define F 0.9              // 缩放因子（较大的F增强探索能力）
#define CR 0.7             // 交叉概率（粒级别，适当降低以平衡探索与开发）

// ========================================
// 早停机制参数设置
// ========================================
#define EARLY_STOP 1       // 早停开关
#define NO_IMPROVE_LIMIT 100 // 无改进代数限制（给予更多耐心）

// ========================================
// 重启机制参数设置
// ========================================
/*
 * 重启机制说明：
 * 当算法陷入局部最优时（长时间无改进），重新初始化部分种群
 * 保留最优个体和部分优秀个体，其余个体重新随机生成
 * 这样既保留了已找到的好解，又增加了种群多样性，帮助跳出局部最优
 */
#define ENABLE_RESTART 1   // 重启机制开关：1=开启，0=关闭
#define RESTART_THRESHOLD 50 // 触发重启的无改进代数阈值
#define KEEP_BEST_RATIO 0.2   // 重启时保留的优秀个体比例（20%）

/*
 * ========================================
 * Schwefel函数
 * ========================================
 * 
 * 数学公式：
 * f(x) = 418.9829 * D - Σ(x[i] * sin(sqrt(|x[i]|)))
 * 
 * 计算步骤：
 * 1. 对每个维度计算 x[i] * sin(sqrt(|x[i]|))
 * 2. 求和
 * 3. 用 418.9829 * D 减去该和
 * 
 * 参数：
 *   x - 长度为DIM的数组，表示一个候选解
 * 返回值：
 *   该候选解的函数值（越小越好，全局最优约为0）
 */
double schwefel_function(double *x) {
    double sum = 0.0;
    
    for (int i = 0; i < DIM; i++) {
        // 计算 x[i] * sin(sqrt(|x[i]|))
        sum += x[i] * sin(sqrt(fabs(x[i])));
    }
    
    // Schwefel函数公式
    double result = 418.9829 * DIM - sum;
    
    return result;
}

/*
 * ========================================
 * 随机数生成函数
 * ========================================
 */
double rand_double() {
    return (double)rand() / RAND_MAX;
}

/*
 * ========================================
 * 种群初始化函数
 * ========================================
 * 
 * 在搜索空间[-500, 500]内随机生成初始种群
 * 
 * 参数：
 *   pop - 二维数组，存储整个种群，大小为POP_SIZE × DIM
 */
void initialize_population(double pop[][DIM]) {
    for (int i = 0; i < POP_SIZE; i++) {
        for (int j = 0; j < DIM; j++) {
            pop[i][j] = LOWER_BOUND + rand_double() * (UPPER_BOUND - LOWER_BOUND);
        }
    }
}

/*
 * ========================================
 * 种群重启函数
 * ========================================
 * 
 * 功能：当算法陷入局部最优时，重新初始化部分种群
 * 
 * 策略：
 * 1. 对种群按适应度排序
 * 2. 保留前KEEP_BEST_RATIO比例的优秀个体
 * 3. 其余个体重新随机初始化
 * 4. 这样既保留了搜索成果，又增加了多样性
 * 
 * 参数：
 *   pop - 种群数组
 *   fitness - 适应度数组
 */
void restart_population(double pop[][DIM], double *fitness) {
    int keep_count = (int)(POP_SIZE * KEEP_BEST_RATIO);
    
    // 简单的选择排序，找出最优的keep_count个个体
    // 将它们移到数组前面
    for (int i = 0; i < keep_count; i++) {
        int best_idx = i;
        for (int j = i + 1; j < POP_SIZE; j++) {
            if (fitness[j] < fitness[best_idx]) {
                best_idx = j;
            }
        }
        
        // 交换个体i和best_idx
        if (best_idx != i) {
            // 交换适应度
            double temp_fitness = fitness[i];
            fitness[i] = fitness[best_idx];
            fitness[best_idx] = temp_fitness;
            
            // 交换个体
            for (int k = 0; k < DIM; k++) {
                double temp = pop[i][k];
                pop[i][k] = pop[best_idx][k];
                pop[best_idx][k] = temp;
            }
        }
    }
    
    // 重新初始化剩余个体
    for (int i = keep_count; i < POP_SIZE; i++) {
        for (int j = 0; j < DIM; j++) {
            pop[i][j] = LOWER_BOUND + rand_double() * (UPPER_BOUND - LOWER_BOUND);
        }
        // 重新计算适应度
        fitness[i] = schwefel_function(pop[i]);
    }
    
    printf(">>> 种群重启：保留前%d个优秀个体，重新初始化其余%d个个体\n", 
           keep_count, POP_SIZE - keep_count);
}

/*
 * ========================================
 * 粒度差分进化算法主函数
 * ========================================
 * 
 * 核心改进：
 * 与传统DE的区别在于变异和交叉操作以"粒"为单位，而非单个维度
 * 
 * 传统DE交叉：
 *   for (j = 0; j < DIM; j++)
 *       if (rand() < CR) trial[j] = mutant[j]
 * 
 * 粒度DE交叉：
 *   for (g = 0; g < N_GRAINS; g++)
 *       if (rand() < CR) 
 *           for (j in grain_g) trial[j] = mutant[j]
 * 
 * 这样可以保持粒内变量的协同关系
 */
void granular_differential_evolution() {
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
        fitness[i] = schwefel_function(population[i]);
        if (fitness[i] < best_fitness) {
            best_fitness = fitness[i];
            for (int j = 0; j < DIM; j++) {
                best_solution[j] = population[i][j];
            }
        }
    }
    
    printf("初始最优值: %.6f\n\n", best_fitness);
    
    // ========================================
    // 步骤3：主循环 - 粒度进化
    // ========================================
    for (int gen = 0; gen < MAX_GEN; gen++) {
        prev_best_fitness = best_fitness;
        
        // 对每个个体进行粒度进化操作
        for (int i = 0; i < POP_SIZE; i++) {
            
            // ========================================
            // 步骤3.1：选择个体用于变异
            // ========================================
            /*
             * 变异策略选择：DE/best/1
             * 
             * 传统DE/rand/1: mutant = x_r1 + F * (x_r2 - x_r3)
             *   - 随机选择基向量，探索能力强但收敛慢
             * 
             * DE/best/1: mutant = x_best + F * (x_r1 - x_r2)
             *   - 使用当前最优解作为基向量，收敛更快
             *   - 对Schwefel这类有明确全局最优的函数效果更好
             */
            int r1, r2;
            do { r1 = rand() % POP_SIZE; } while (r1 == i);
            do { r2 = rand() % POP_SIZE; } while (r2 == i || r2 == r1);
            
            // ========================================
            // 步骤3.2：粒度变异操作（使用DE/best/1策略）
            // ========================================
            double mutant[DIM];
            for (int j = 0; j < DIM; j++) {
                // DE/best/1: 以最优解为基础进行变异
                mutant[j] = best_solution[j] + F * (population[r1][j] - population[r2][j]);
            }
            
            // ========================================
            // 步骤3.3：粒度交叉操作（核心改进）
            // ========================================
            /*
             * 关键区别：以粒为单位进行交叉决策
             * 
             * 传统DE：每个维度独立决定是否交叉
             *   - 可能导致粒内部分维度来自变异向量，部分来自目标向量
             *   - 破坏了粒内变量的协同关系
             * 
             * 粒度DE：整个粒一起决定是否交叉
             *   - 保证粒内所有维度要么全部来自变异向量，要么全部来自目标向量
             *   - 保护了粒内变量的协同关系
             */
            
            // 随机选择一个粒，确保至少有一个粒被交叉
            int g_rand = rand() % N_GRAINS;
            
            // 对每个粒进行交叉决策
            for (int g = 0; g < N_GRAINS; g++) {
                // 计算当前粒包含的维度范围
                int start_dim = g * GRAIN_SIZE;      // 粒的起始维度
                int end_dim = start_dim + GRAIN_SIZE; // 粒的结束维度
                
                // 以概率CR决定是否交叉整个粒，或者是被强制选中的粒
                if (rand_double() < CR || g == g_rand) {
                    // 交叉：整个粒的所有维度都使用变异向量
                    for (int j = start_dim; j < end_dim; j++) {
                        trial[j] = mutant[j];
                        
                        // 边界处理
                        if (trial[j] < LOWER_BOUND) trial[j] = LOWER_BOUND;
                        if (trial[j] > UPPER_BOUND) trial[j] = UPPER_BOUND;
                    }
                } else {
                    // 不交叉：整个粒的所有维度都保留目标向量
                    for (int j = start_dim; j < end_dim; j++) {
                        trial[j] = population[i][j];
                    }
                }
            }
            
            // ========================================
            // 步骤3.4：选择操作
            // ========================================
            double trial_fitness = schwefel_function(trial);
            
            if (trial_fitness < fitness[i]) {
                // 接受试验向量
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
        
        // ========================================
        // 重启机制判断
        // ========================================
        /*
         * 重启触发条件：
         * 1. 重启机制已开启
         * 2. 连续无改进代数达到重启阈值
         * 3. 还未达到早停限制（给重启后的种群机会继续进化）
         * 4. 不在最后几代（避免频繁重启）
         */
        if (ENABLE_RESTART && 
            no_improve_count >= RESTART_THRESHOLD && 
            no_improve_count < NO_IMPROVE_LIMIT &&
            gen < MAX_GEN - 50) {
            
            printf("\n>>> 检测到陷入局部最优（连续%d代无改进）\n", no_improve_count);
            restart_population(population, fitness);
            no_improve_count = 0;  // 重置计数器
            printf(">>> 重启完成，继续进化...\n\n");
        }
        
        // 每20代输出一次结果
        if ((gen + 1) % 50 == 0) {
            printf("第 %d 代, 最优值: %.6f", gen + 1, best_fitness);
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
    
    printf("\n最优解的各维度值：\n");
    printf("维度\t\t值\t\t所属粒\n");
    printf("----------------------------------------\n");
    for (int i = 0; i < DIM; i++) {
        int grain_id = i / GRAIN_SIZE;
        printf("x[%d]\t\t%.6f\t粒%d\n", i, best_solution[i], grain_id);
    }
    
    printf("\n----------------------------------------\n");
    printf("最优值: %.6f\n", best_fitness);
    printf("理论最优值: 0.0 (在 x[i] = 420.9687 处)\n");
    printf("误差: %.6f\n", best_fitness);
    
    // 计算解与理论最优解的距离
    double distance = 0.0;
    for (int i = 0; i < DIM; i++) {
        double diff = best_solution[i] - 420.9687;
        distance += diff * diff;
    }
    distance = sqrt(distance);
    printf("解的欧氏距离: %.6f\n", distance);
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
    printf("粒度差分进化算法求解Schwefel函数\n");
    printf("========================================\n");
    printf("问题维度: %d\n", DIM);
    printf("搜索空间: [%.1f, %.1f]\n", LOWER_BOUND, UPPER_BOUND);
    printf("粒大小: %d 维/粒\n", GRAIN_SIZE);
    printf("粒数量: %d 个粒\n", N_GRAINS);
    printf("种群大小: %d\n", POP_SIZE);
    printf("最大迭代次数: %d\n", MAX_GEN);
    if (EARLY_STOP) {
        printf("早停机制: 开启 (连续%d代无改进则终止)\n", NO_IMPROVE_LIMIT);
    } else {
        printf("早停机制: 关闭\n");
    }
    if (ENABLE_RESTART) {
        printf("重启机制: 开启 (连续%d代无改进触发重启)\n", RESTART_THRESHOLD);
        printf("重启策略: 保留前%.0f%%优秀个体\n", KEEP_BEST_RATIO * 100);
    } else {
        printf("重启机制: 关闭\n");
    }
    printf("========================================\n\n");
    
    printf("粒划分方案：\n");
    for (int g = 0; g < N_GRAINS; g++) {
        int start = g * GRAIN_SIZE;
        int end = start + GRAIN_SIZE - 1;
        printf("粒%d: 维度 [%d, %d]\n", g, start, end);
    }
    printf("========================================\n\n");
    
    granular_differential_evolution();
    
    printf("\n========================================\n");
    printf("算法说明：\n");
    printf("粒度DE与传统DE的区别：\n");
    printf("- 传统DE: 逐维独立交叉，可能破坏变量关联\n");
    printf("- 粒度DE: 以粒为单位交叉，保护变量协同关系\n");
    printf("- 变异策略: DE/best/1，加速向全局最优收敛\n");
    printf("- 重启机制: 检测到局部最优时重启部分种群\n");
    printf("- 优势: 减少盲目破坏，实现粗-细两层搜索\n");
    printf("\n提示：\n");
    printf("- Schwefel函数是高度多峰函数，优化难度大\n");
    printf("- 重启机制帮助算法跳出局部最优\n");
    printf("- 如果结果不理想，可以多运行几次\n");
    printf("- 调整GRAIN_SIZE可以改变粒的划分策略\n");
    printf("- 可以通过修改ENABLE_RESTART开关重启机制\n");
    printf("========================================\n");
    
    return 0;
}
