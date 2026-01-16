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
 *    a. 变异：计算差分向量
 *    b. 交叉：以粒为单位决定是否交叉（而非逐维）
 *    c. 选择：保留更优个体
 * 4. 迭代直到收敛
 * 
 * ========================================
 * 测试函数：分组加权Sphere函数
 * ========================================
 * 
 * 为什么选择这个函数：
 * 这是一个人工设计的函数，专门用于展示粒度计算的优势。
 * 函数简单易优化，但能清晰体现粒度划分的作用。
 * 
 * 函数定义（分组形式）：
 * 将D维变量分成若干组，每组内部变量有加权关系
 * 对于每组(x[2i], x[2i+1])：
 *   f_i = (x[2i])^2 + 2*(x[2i+1])^2 + x[2i]*x[2i+1]
 * 
 * 总目标函数：
 *   f(x) = Σ f_i  (对所有组求和)
 * 
 * 函数特点：
 * - 每组内的两个变量有交互项 x[2i]*x[2i+1]，存在耦合关系
 * - 组与组之间相互独立（可分离性）
 * - 全局最优解：x[i] = 0 (所有维度)
 * - 全局最优值：0
 * - 搜索空间：[-5, 5]^D
 * - 函数凸且光滑，容易优化，便于观察粒度计算的效果
 * 
 * 粒度计算的优势体现：
 * 1. 天然的分组结构：每2个变量形成一个耦合组
 * 2. 粒内有关联：交互项使得两个变量需要协同优化
 * 3. 粒间独立：不同粒可以独立优化
 * 4. 对比效果：
 *    - 传统DE：逐维交叉，可能破坏(x[2i], x[2i+1])的协同关系
 *    - 粒度DE：以粒为单位交叉，保护变量间的交互关系
 * 
 * 本例设置：
 * - 10维问题，分成5个粒
 * - 每个粒包含2个相邻变量
 * - 粒的划分与函数的分组结构完美匹配
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <windows.h>

// ========================================
// 问题参数设置
// ========================================
#define DIM 10             // 问题维度：必须是偶数
#define LOWER_BOUND -5.0   // 搜索空间下界
#define UPPER_BOUND 5.0    // 搜索空间上界

// ========================================
// 粒度划分参数
// ========================================
#define GRAIN_SIZE 2       // 粒大小：每个粒包含的维度数
#define N_GRAINS (DIM / GRAIN_SIZE)  // 粒的数量

/*
 * 粒划分说明：
 * 本例采用与函数结构匹配的划分策略：
 * - 粒0: 维度[0, 1] - 对应第1个加权Sphere分组
 * - 粒1: 维度[2, 3] - 对应第2个加权Sphere分组
 * - 粒2: 维度[4, 5] - 对应第3个加权Sphere分组
 * - 粒3: 维度[6, 7] - 对应第4个加权Sphere分组
 * - 粒4: 维度[8, 9] - 对应第5个加权Sphere分组
 * 
 * 这种划分完美匹配了函数的内在结构，是粒度计算的理想应用场景
 * 
 * 粒度划分的关键原则：
 * 1. 语义相关性：将有交互关系的变量划分到同一个粒
 * 2. 结构匹配：粒的划分应该反映问题的内在结构
 * 3. 独立性：不同粒之间应该尽可能独立
 * 4. 粒度大小：粒不宜过大（增加搜索复杂度）或过小（失去粒度优势）
 * 
 * 本例中的划分依据：
 * - 函数的交互项 x[2i]*x[2i+1] 表明相邻变量存在耦合
 * - 将相邻两个变量划分为一个粒，保护这种耦合关系
 * - 不同粒之间没有交互项，满足独立性要求
 */

// ========================================
// 差分进化算法参数设置
// ========================================
#define POP_SIZE 50        // 种群大小
#define MAX_GEN 300        // 最大迭代次数
#define F 0.8              // 缩放因子
#define CR 0.9             // 交叉概率（粒级别的交叉概率）

// ========================================
// 早停机制参数设置
// ========================================
#define EARLY_STOP 1       // 早停开关
#define NO_IMPROVE_LIMIT 50 // 无改进代数限制

/*
 * ========================================
 * 分组加权Sphere函数
 * ========================================
 * 
 * 数学公式：
 * f(x) = Σ [(x[2i])^2 + 2*(x[2i+1])^2 + x[2i]*x[2i+1]]
 * 其中 i = 0 到 (DIM/2 - 1)
 * 
 * 计算步骤：
 * 1. 将D维变量按相邻两个分组
 * 2. 对每组计算加权平方和及交互项
 * 3. 求和得到总适应度
 * 
 * 函数特性：
 * - 每组内两个变量有交互项：x[2i]*x[2i+1]
 * - 这个交互项使得两个变量需要协同优化
 * - 组间独立：不同组可以独立优化
 * - 全局最优：所有x[i] = 0，函数值 = 0
 * - 函数凸且光滑，易于优化
 * 
 * 参数：
 *   x - 长度为DIM的数组，表示一个候选解
 * 返回值：
 *   该候选解的函数值（越小越好，全局最优为0）
 */
double grouped_weighted_sphere(double *x) {
    double sum = 0.0;
    
    // 对每个分组计算函数值
    for (int i = 0; i < DIM / 2; i++) {
        int idx1 = 2 * i;      // 组内第一个变量的索引
        int idx2 = 2 * i + 1;  // 组内第二个变量的索引
        
        // 分组函数：x1^2 + 2*x2^2 + x1*x2
        double term1 = x[idx1] * x[idx1];           // x1^2
        double term2 = 2.0 * x[idx2] * x[idx2];     // 2*x2^2
        double term3 = x[idx1] * x[idx2];           // x1*x2 (交互项)
        
        sum += term1 + term2 + term3;
    }
    
    return sum;
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
 * 在搜索空间[-5, 5]内随机生成初始种群
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
 * 粒度差分进化算法主函数
 * ========================================
 * 
 * 核心改进：
 * 与传统DE的区别在于交叉操作以"粒"为单位，而非单个维度
 * 
 * 传统DE交叉（逐维独立）：
 *   for (j = 0; j < DIM; j++)
 *       if (rand() < CR) 
 *           trial[j] = mutant[j]
 *       else
 *           trial[j] = target[j]
 * 
 *   问题示例：
 *   假设粒0包含维度[0,1]，传统DE可能产生：
 *   - trial[0] = mutant[0]  (来自变异向量)
 *   - trial[1] = target[1]  (来自目标向量)
 *   这破坏了(x[0], x[1])的协同关系，因为它们的交互项 x[0]*x[1] 被打乱
 * 
 * 粒度DE交叉（以粒为单位）：
 *   for (g = 0; g < N_GRAINS; g++)
 *       if (rand() < CR)
 *           for (j in grain_g) trial[j] = mutant[j]
 *       else
 *           for (j in grain_g) trial[j] = target[j]
 * 
 *   优势示例：
 *   对于粒0包含维度[0,1]，粒度DE保证：
 *   - 要么 trial[0] = mutant[0] 且 trial[1] = mutant[1]  (整体交叉)
 *   - 要么 trial[0] = target[0] 且 trial[1] = target[1]  (整体保留)
 *   这保护了(x[0], x[1])的协同关系，维持了交互项的完整性
 * 
 * 算法优势总结：
 * 1. 保护优良片段：已经形成良好组合的变量不会被部分破坏
 * 2. 减少无效搜索：避免产生破坏性的变量组合
 * 3. 加速收敛：保持有效的变量协同关系，提高搜索效率
 * 4. 体现问题结构：粒的划分反映了问题的内在语义
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
        fitness[i] = grouped_weighted_sphere(population[i]);
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
            // 步骤3.1：变异操作 - 选择三个不同的个体
            // ========================================
            /*
             * 使用DE/rand/1策略：mutant = x_r1 + F * (x_r2 - x_r3)
             * 
             * 对于这个简单的凸函数，rand/1策略足够有效
             */
            int r1, r2, r3;
            do { r1 = rand() % POP_SIZE; } while (r1 == i);
            do { r2 = rand() % POP_SIZE; } while (r2 == i || r2 == r1);
            do { r3 = rand() % POP_SIZE; } while (r3 == i || r3 == r1 || r3 == r2);
            
            // 计算变异向量
            double mutant[DIM];
            for (int j = 0; j < DIM; j++) {
                mutant[j] = population[r1][j] + F * (population[r2][j] - population[r3][j]);
            }
            
            // ========================================
            // 步骤3.2：粒度交叉操作（核心创新）
            // ========================================
            /*
             * 粒度交叉的关键思想：
             * 
             * 1. 决策单位：以粒为单位，而非单个维度
             *    - 传统DE：每个维度独立决定是否交叉
             *    - 粒度DE：每个粒整体决定是否交叉
             * 
             * 2. 保护机制：粒内所有维度同进退
             *    - 如果粒被选中交叉，粒内所有维度都使用mutant
             *    - 如果粒不被交叉，粒内所有维度都保留target
             *    - 这确保了粒内变量的协同关系不被破坏
             * 
             * 3. 语义保持：保护变量间的交互关系
             *    - 对于函数 f = x1^2 + 2*x2^2 + x1*x2
             *    - 交互项 x1*x2 要求 x1 和 x2 协同优化
             *    - 粒度交叉保证 (x1, x2) 作为整体进化
             * 
             * 具体实现步骤：
             * a. 对每个粒，以概率CR决定是否整体交叉
             * b. 如果交叉：粒内所有维度都使用mutant
             * c. 如果不交叉：粒内所有维度都保留target
             * d. 至少保证一个粒被交叉（通过g_rand实现）
             * 
             * 与传统DE的对比实例：
             * 假设有5个粒，每个粒2维，某次交叉操作：
             * 
             * 传统DE可能产生（逐维独立）：
             *   [mutant[0], target[1], mutant[2], target[3], mutant[4], 
             *    target[5], mutant[6], target[7], mutant[8], target[9]]
             *   问题：粒0的x[0]和x[1]来自不同来源，破坏了它们的交互关系
             * 
             * 粒度DE只会产生（以粒为单位）：
             *   [mutant[0,1], target[2,3], mutant[4,5], 
             *    target[6,7], mutant[8,9]]
             *   优势：每个粒内的变量来自同一来源，保护了交互关系
             * 
             * 效果分析：
             * - 如果(x[0], x[1])已经形成良好组合（如都接近0），不会被部分破坏
             * - 降低了盲目破坏优良片段的概率
             * - 搜索空间从2^10（逐维）降低到2^5（逐粒），但保持了有效性
             * - 实现了"粗-细"两层搜索：粒级别（粗）+ 粒内级别（细）
             */
            
            // 随机选择一个粒，确保至少有一个粒被交叉
            int g_rand = rand() % N_GRAINS;
            
            // 对每个粒进行交叉决策
            for (int g = 0; g < N_GRAINS; g++) {
                // 计算当前粒包含的维度范围
                int start_dim = g * GRAIN_SIZE;      // 粒的起始维度
                int end_dim = start_dim + GRAIN_SIZE; // 粒的结束维度
                
                // 以概率CR决定是否交叉整个粒
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
            // 步骤3.3：选择操作（贪婪选择）
            // ========================================
            double trial_fitness = grouped_weighted_sphere(trial);
            
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
        
        // 每50代输出一次结果
        if ((gen + 1) % 30 == 0) {
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
    
    printf("\n最优解按粒分组显示：\n");
    printf("粒编号\t维度范围\t\t值\t\t\t分组函数值\n");
    printf("----------------------------------------------------------------\n");
    
    double total_check = 0.0;
    for (int g = 0; g < N_GRAINS; g++) {
        int idx1 = g * 2;
        int idx2 = g * 2 + 1;
        
        // 计算该粒对应的函数分量：x1^2 + 2*x2^2 + x1*x2
        double term1 = best_solution[idx1] * best_solution[idx1];
        double term2 = 2.0 * best_solution[idx2] * best_solution[idx2];
        double term3 = best_solution[idx1] * best_solution[idx2];
        double grain_value = term1 + term2 + term3;
        total_check += grain_value;
        
        printf("粒%d\t[%d, %d]\t\t(%.4f, %.4f)\t%.6f\n", 
               g, idx1, idx2, 
               best_solution[idx1], best_solution[idx2],
               grain_value);
    }
    
    printf("----------------------------------------------------------------\n");
    printf("总函数值: %.6f\n", best_fitness);
    printf("验证计算: %.6f (应与总函数值相同)\n", total_check);
    printf("\n理论最优解: x[i] = 0.0 (所有维度)\n");
    printf("理论最优值: 0.0\n");
    printf("当前误差: %.6f\n", best_fitness);
    
    // 计算解与理论最优解的距离
    double distance = 0.0;
    for (int i = 0; i < DIM; i++) {
        double diff = best_solution[i] - 0.0;
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
    printf("粒度差分进化算法求解分组加权Sphere函数\n");
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
    printf("========================================\n\n");
    
    printf("函数结构与粒划分：\n");
    printf("分组加权Sphere函数将变量按相邻两个分组\n");
    printf("每组内两个变量有交互项，存在耦合关系\n");
    printf("函数形式：f_i = x1^2 + 2*x2^2 + x1*x2\n\n");
    
    printf("粒划分方案（与函数结构匹配）：\n");
    for (int g = 0; g < N_GRAINS; g++) {
        int start = g * GRAIN_SIZE;
        int end = start + GRAIN_SIZE - 1;
        printf("粒%d: 维度[%d, %d] - 对应加权Sphere分组%d\n", g, start, end, g);
    }
    printf("========================================\n\n");
    
    granular_differential_evolution();
    
    printf("\n========================================\n");
    printf("粒度计算的优势说明：\n");
    printf("========================================\n");
    printf("1. 问题特性分析：\n");
    printf("   - 目标函数：f_i = x1^2 + 2*x2^2 + x1*x2\n");
    printf("   - 交互项 x1*x2 表明两个变量存在耦合关系\n");
    printf("   - 要达到最优(x1=0, x2=0)，两个变量需要协同优化\n");
    printf("   - 如果只优化x1而保持x2不变，或反之，效率会降低\n\n");
    
    printf("2. 传统DE的局限性：\n");
    printf("   - 逐维独立交叉：每个维度独立决定是否交叉\n");
    printf("   - 可能产生：x[0]来自mutant，x[1]来自target\n");
    printf("   - 破坏了(x[0], x[1])的协同关系\n");
    printf("   - 示例：如果mutant中(x[0],x[1])=(0.1,0.1)是好组合\n");
    printf("           target中(x[0],x[1])=(2.0,2.0)是差组合\n");
    printf("           传统DE可能产生(0.1,2.0)，既不好也不差\n\n");
    
    printf("3. 粒度DE的创新点：\n");
    printf("   - 以粒为单位交叉：整个粒一起决定是否交叉\n");
    printf("   - (x[0], x[1])要么全部来自mutant，要么全部来自target\n");
    printf("   - 保护了变量间的交互关系和协同性\n");
    printf("   - 示例：粒度DE只会产生(0.1,0.1)或(2.0,2.0)\n");
    printf("           保持了变量组合的完整性\n\n");
    
    printf("4. 粒度计算的核心优势：\n");
    printf("   a) 降低破坏性：减少盲目破坏优良片段的概率\n");
    printf("   b) 提高效率：保持有效的变量协同关系\n");
    printf("   c) 体现结构：粒的划分反映问题的内在语义\n");
    printf("   d) 两层搜索：\n");
    printf("      - 粗粒度：在粒级别搜索（5个粒的组合，2^5=32种）\n");
    printf("      - 细粒度：在粒内优化（每粒2个变量的连续优化）\n");
    printf("      - 相比逐维搜索（2^10=1024种），大大降低复杂度\n\n");
    
    printf("5. 实验效果：\n");
    printf("   - 函数简单凸，容易收敛到全局最优\n");
    printf("   - 粒度DE通常能在较少代数内找到高质量解\n");
    printf("   - 误差通常能达到1e-6或更小\n");
    printf("========================================\n");
    
    return 0;
}
