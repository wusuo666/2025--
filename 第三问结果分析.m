% 第三问结果分析：执行100次PSO优化并统计结果
% 使用并行计算池加速（20核）

clear; clc;

%% 开启并行计算池
% 设置目标核心数
target_cores = 20;

try
    % 检查并行配置
    c = parcluster('local');
    disp(['当前默认最大工作进程数: ', num2str(c.NumWorkers)]);
    
    % 修改最大工作进程数
    c.NumWorkers = target_cores;
    saveProfile(c);
    disp(['已修改最大工作进程数为: ', num2str(c.NumWorkers)]);
    
    % 检查是否已存在并行池
    poolobj = gcp('nocreate');
    if isempty(poolobj)
        % 创建新的并行池
        parpool(c, target_cores);
        poolobj = gcp;
        disp(['成功创建并行计算池，使用 ', num2str(poolobj.NumWorkers), ' 个核心']);
    else
        disp(['已存在并行池，当前使用 ', num2str(poolobj.NumWorkers), ' 个核心']);
        % 如果需要强制使用目标数量的核心，先删除现有池再创建新池
        if poolobj.NumWorkers ~= target_cores
            disp('删除现有并行池以调整核心数...');
            delete(poolobj);
            parpool(c, target_cores);
            poolobj = gcp;
            disp(['调整并行计算池至 ', num2str(poolobj.NumWorkers), ' 个核心']);
        end
    end
    
    % 打印实际使用的核心数
    fprintf('实际使用的核心数: %d\n', poolobj.NumWorkers);
    
catch err
    warning('创建并行池时出错: %s', err.message);
    disp('将尝试使用最大可用核心数');
    try
        parpool('local');
        poolobj = gcp;
        disp(['使用默认配置创建并行池，核心数: ', num2str(poolobj.NumWorkers)]);
    catch
        warning('无法创建并行池，将使用串行计算');
    end
end

%% 参数初始化
num_runs = 100;  % 优化重复次数

% 保存每次运行的结果
all_best_durations = zeros(num_runs, 1);  % 最佳遮挡时长
all_best_positions = zeros(num_runs, 8);  % 最佳参数 [s, theta, t1, dt1, t2, dt2, t3, dt3]
all_best_rewards = zeros(num_runs, 1);    % 最佳奖赏值
all_iterations = zeros(num_runs, 1);      % 收敛所需迭代次数
all_convergence = cell(num_runs, 1);      % 收敛曲线数据

% 优化参数（与Question3.m保持一致）
max_iterations = 60;  % 最大迭代次数

%% 并行执行100次优化
disp('开始并行执行100次优化...');
tic;  % 开始计时

parfor run_idx = 1:num_runs
    % 调用优化函数
    [best_position, best_duration, best_reward, convergence, actual_iterations] = run_pso_optimization();
    
    % 存储结果
    all_best_durations(run_idx) = best_duration;
    all_best_positions(run_idx, :) = best_position;
    all_best_rewards(run_idx) = best_reward;
    all_iterations(run_idx) = actual_iterations;
    all_convergence{run_idx} = convergence;
    
    % 简短的进度更新（仅从主工作线程显示，所以可能不会显示所有进度）
    if mod(run_idx, 10) == 0
        fprintf('完成第 %d/%d 次优化\n', run_idx, num_runs);
    end
end

runtime = toc;  % 结束计时
fprintf('100次优化完成，总耗时: %.2f 秒\n', runtime);

%% 结果统计与分析
% 找出最佳结果
[max_duration, max_idx] = max(all_best_durations);
best_of_all = all_best_positions(max_idx, :);
s_best = best_of_all(1);
th_best = best_of_all(2);
t1_best = best_of_all(3);
dt1_best = best_of_all(4);
t2_best = best_of_all(5);
dt2_best = best_of_all(6);
t3_best = best_of_all(7);
dt3_best = best_of_all(8);
best_reward = all_best_rewards(max_idx);

vx_best = s_best * cos(th_best);
vy_best = s_best * sin(th_best);

% 计算统计指标
mean_duration = mean(all_best_durations);
std_duration = std(all_best_durations);
median_duration = median(all_best_durations);
min_duration = min(all_best_durations);
q1_duration = quantile(all_best_durations, 0.25);
q3_duration = quantile(all_best_durations, 0.75);

% 平均收敛曲线（可能长度不一，需处理）
max_len = max(cellfun(@length, all_convergence));
padded_curves = zeros(num_runs, max_len);
for i = 1:num_runs
    curve = all_convergence{i};
    padded_curves(i, 1:length(curve)) = curve;
    % 对于长度不足的，用最后一个值填充
    if length(curve) < max_len
        padded_curves(i, length(curve)+1:end) = curve(end);
    end
end
mean_convergence = mean(padded_curves, 1);

%% 保存结果到MAT文件
% 创建结果结构体
results = struct();
results.best_durations = all_best_durations;
results.best_positions = all_best_positions;
results.best_rewards = all_best_rewards;
results.iterations = all_iterations;
results.convergence = all_convergence;
results.statistics.max_duration = max_duration;
results.statistics.mean_duration = mean_duration;
results.statistics.std_duration = std_duration;
results.statistics.median_duration = median_duration;
results.statistics.q1_duration = q1_duration;
results.statistics.q3_duration = q3_duration;
results.global_best.position = best_of_all;
results.global_best.duration = max_duration;
results.global_best.reward = best_reward;
results.global_best.vx = vx_best;
results.global_best.vy = vy_best;
results.mean_convergence = mean_convergence;
results.runtime = runtime;

% 生成带时间戳的文件名
timestamp = datestr(now, 'yyyymmdd_HHMMSS');
filename = sprintf('PSO_Q3_Results_%s.mat', timestamp);

% 保存结果
save(filename, 'results');
fprintf('结果已保存到文件: %s\n', filename);

%% 结果输出
fprintf('\n========== 结果统计 ==========\n');
fprintf('最佳遮挡时长: %.6f 秒\n', max_duration);
fprintf('最佳奖赏值: %.6f\n', best_reward);
fprintf('平均遮挡时长: %.6f 秒 (std: %.6f)\n', mean_duration, std_duration);
fprintf('最短遮挡时长: %.6f 秒\n', min_duration);
fprintf('中位数遮挡时长: %.6f 秒\n', median_duration);
fprintf('四分位数 Q1/Q3: %.6f / %.6f 秒\n', q1_duration, q3_duration);
fprintf('平均收敛迭代数: %.2f\n', mean(all_iterations));

fprintf('\n========== 全局最优解 ==========\n');
fprintf('  s (速度大小) = %.6f m/s\n', s_best);
fprintf('  theta (方位角) = %.6f rad (%.2f度)\n', th_best, rad2deg(th_best));
fprintf('  vx_FY1 = %.6f m/s\n', vx_best);
fprintf('  vy_FY1 = %.6f m/s\n', vy_best);
fprintf('\n  第一次投弹：\n');
fprintf('    t_throw1 = %.6f s\n', t1_best);
fprintf('    t_explode1 = %.6f s\n', dt1_best);
fprintf('\n  第二次投弹：\n');
fprintf('    t_throw2 = %.6f s\n', t2_best);
fprintf('    t_explode2 = %.6f s\n', dt2_best);
fprintf('\n  第三次投弹：\n');
fprintf('    t_throw3 = %.6f s\n', t3_best);
fprintf('    t_explode3 = %.6f s\n', dt3_best);

%% 可视化结果
% 保存图表设置
save_figures = true;  % 是否保存图表到文件
fig_format = 'png';   % 图表保存格式（'png', 'jpg', 'fig'等）
fig_folder = 'PSO_Q3_Figures';  % 图表保存文件夹

% 创建图表保存文件夹
if save_figures
    if ~exist(fig_folder, 'dir')
        mkdir(fig_folder);
        fprintf('创建图表保存文件夹: %s\n', fig_folder);
    end
end

% 1. 收敛曲线
figure('Name', '平均收敛曲线', 'Position', [100, 100, 800, 500]);
plot(1:length(mean_convergence), -mean_convergence, 'LineWidth', 2);
grid on; box on;
xlabel('迭代次数'); ylabel('平均最佳遮挡时长 (秒)');
title('100次PSO优化的平均收敛曲线');

% 保存图表
if save_figures
    fig_file = fullfile(fig_folder, sprintf('PSO_Q3_收敛曲线_%s.%s', timestamp, fig_format));
    saveas(gcf, fig_file);
    fprintf('收敛曲线已保存: %s\n', fig_file);
end

% 2. 结果分布直方图
figure('Name', '遮挡时长分布', 'Position', [100, 600, 800, 500]);
histogram(all_best_durations, 20, 'Normalization', 'count');
hold on;
line([mean_duration, mean_duration], ylim, 'Color', 'r', 'LineWidth', 2, 'LineStyle', '--');
line([max_duration, max_duration], ylim, 'Color', 'g', 'LineWidth', 2, 'LineStyle', '--');
grid on; box on;
xlabel('最佳遮挡时长 (秒)'); ylabel('频数');
title('100次PSO优化的遮挡时长分布');
legend('频率分布', '平均值', '最大值');

% 保存图表
if save_figures
    fig_file = fullfile(fig_folder, sprintf('PSO_Q3_遮挡时长分布_%s.%s', timestamp, fig_format));
    saveas(gcf, fig_file);
    fprintf('遮挡时长分布图已保存: %s\n', fig_file);
end

% 3. 参数散点图 - 适用于第三问的多参数
figure('Name', '参数散点图', 'Position', [900, 100, 1000, 800]);

% s 和 theta 的散点图
subplot(2,2,1);
scatter(all_best_positions(:,1), all_best_positions(:,2), 50, all_best_durations, 'filled');
colorbar; grid on;
xlabel('速度大小 s (m/s)'); ylabel('方位角 theta (rad)');
title('速度参数与遮挡时长关系');
colormap(jet);

% 投弹时间散点图 (t1, t2, t3)
subplot(2,2,2);
scatter3(all_best_positions(:,3), all_best_positions(:,5), all_best_positions(:,7), 50, all_best_durations, 'filled');
colorbar; grid on;
xlabel('t1 (s)'); ylabel('t2 (s)'); zlabel('t3 (s)');
title('三次投弹时间与遮挡时长关系');
view(45, 30);

% 引爆延迟时间散点图 (dt1, dt2, dt3)
subplot(2,2,3);
scatter3(all_best_positions(:,4), all_best_positions(:,6), all_best_positions(:,8), 50, all_best_durations, 'filled');
colorbar; grid on;
xlabel('dt1 (s)'); ylabel('dt2 (s)'); zlabel('dt3 (s)');
title('引爆延迟时间与遮挡时长关系');
view(45, 30);

% 遮挡时长与迭代次数的关系
subplot(2,2,4);
scatter(all_iterations, all_best_durations, 50, all_best_rewards, 'filled');
colorbar; grid on;
xlabel('收敛迭代次数'); ylabel('遮挡时长 (s)');
title('收敛速度、遮挡时长与奖赏关系');

% 保存图表
if save_figures
    fig_file = fullfile(fig_folder, sprintf('PSO_Q3_参数散点图_%s.%s', timestamp, fig_format));
    saveas(gcf, fig_file);
    fprintf('参数散点图已保存: %s\n', fig_file);
end

% 4. 投弹时刻与奖赏值分析
figure('Name', '投弹时刻与奖赏分析', 'Position', [100, 100, 1000, 400]);

% 左图：三次投弹时刻间隔分析
subplot(1,2,1);
interval12 = all_best_positions(:,5) - all_best_positions(:,3);  % t2 - t1
interval23 = all_best_positions(:,7) - all_best_positions(:,5);  % t3 - t2
scatter(interval12, interval23, 50, all_best_durations, 'filled');
colorbar; grid on;
xlabel('间隔1 (t2-t1) (s)'); ylabel('间隔2 (t3-t2) (s)');
title('投弹间隔与遮挡时长关系');

% 右图：奖赏值与遮挡时长关系
subplot(1,2,2);
scatter(all_best_rewards, all_best_durations, 50, 'filled');
grid on;
xlabel('奖赏值'); ylabel('遮挡时长 (s)');
title('奖赏与遮挡时长关系');

% 保存图表
if save_figures
    fig_file = fullfile(fig_folder, sprintf('PSO_Q3_时刻与奖赏分析_%s.%s', timestamp, fig_format));
    saveas(gcf, fig_file);
    fprintf('时刻与奖赏分析图已保存: %s\n', fig_file);
end

% 额外保存一份总结报告
if save_figures
    % 创建文本文件保存统计结果
    report_file = fullfile(fig_folder, sprintf('PSO_Q3_统计报告_%s.txt', timestamp));
    fid = fopen(report_file, 'w');
    if fid ~= -1
        fprintf(fid, '==========================================\n');
        fprintf(fid, '  第三问PSO优化结果统计报告 (%s)\n', timestamp);
        fprintf(fid, '==========================================\n\n');
        fprintf(fid, '优化运行次数: %d\n', num_runs);
        fprintf(fid, '总计算时间: %.2f 秒\n\n', runtime);
        fprintf(fid, '---------- 遮挡时长统计 ----------\n');
        fprintf(fid, '最佳遮挡时长: %.6f 秒\n', max_duration);
        fprintf(fid, '平均遮挡时长: %.6f 秒 (标准差: %.6f)\n', mean_duration, std_duration);
        fprintf(fid, '最短遮挡时长: %.6f 秒\n', min_duration);
        fprintf(fid, '中位数遮挡时长: %.6f 秒\n', median_duration);
        fprintf(fid, '四分位数 Q1/Q3: %.6f / %.6f 秒\n\n', q1_duration, q3_duration);
        fprintf(fid, '---------- 全局最优解 ----------\n');
        fprintf(fid, '速度大小 s = %.6f m/s\n', s_best);
        fprintf(fid, '方位角 theta = %.6f rad (%.2f度)\n', th_best, rad2deg(th_best));
        fprintf(fid, '水平速度 vx = %.6f m/s\n', vx_best);
        fprintf(fid, '水平速度 vy = %.6f m/s\n', vy_best);
        fprintf(fid, '第一次投弹时刻 t1 = %.6f s\n', t1_best);
        fprintf(fid, '第一次引爆延迟 dt1 = %.6f s\n', dt1_best);
        fprintf(fid, '第二次投弹时刻 t2 = %.6f s\n', t2_best);
        fprintf(fid, '第二次引爆延迟 dt2 = %.6f s\n', dt2_best);
        fprintf(fid, '第三次投弹时刻 t3 = %.6f s\n', t3_best);
        fprintf(fid, '第三次引爆延迟 dt3 = %.6f s\n', dt3_best);
        fprintf(fid, '最佳奖赏值 = %.6f\n\n', best_reward);
        fprintf(fid, '---------- 收敛性能 ----------\n');
        fprintf(fid, '平均收敛迭代数: %.2f\n', mean(all_iterations));
        fprintf(fid, '最快收敛迭代数: %d\n', min(all_iterations));
        fprintf(fid, '最慢收敛迭代数: %d\n\n', max(all_iterations));
        fprintf(fid, '结果数据文件: %s\n', filename);
        fclose(fid);
        fprintf('统计报告已保存: %s\n', report_file);
    end
end

%% 并行优化函数
function [best_position, best_duration, best_reward, convergence_curve, actual_iterations] = run_pso_optimization()
    % 约束与场景参数（与Question3.m保持一致）
    SPEED_MIN = 70; SPEED_MAX = 140;
    THETA_MIN = -pi; THETA_MAX = pi;
    VM_kappa = 10;
    T_MAX = 20;
    T_SUM_MAX = 20 * (1 - 1e-6);
    MIN_INTERVAL = 1;
    q1_opts = struct('tSimEnd', 20, 'g', 9.8);
    
    % 奖赏机制参数
    REWARD_MAX_ITER_FRAC = 0.5;
    REWARD_WEIGHT_INIT = 1;
    
    % 粒子群参数
    nVar = 8;  % [s, theta, t1, dt1, t2, dt2, t3, dt3]
    VarMin = [SPEED_MIN, THETA_MIN, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    VarMax = [SPEED_MAX, THETA_MAX, T_MAX, T_MAX, T_MAX, T_MAX, T_MAX, T_MAX];
    
    nPop = 120;
    MaxIt = 60;
    w = 1.2;
    wDamp = 0.99;
    c1 = 1.6;
    c2 = 2.0;
    
    VelMax = 0.25 * (VarMax - VarMin);
    VelMin = -VelMax;
    
    % 目标函数
    obj = @(x, it) objective_function(x, q1_opts, it, MaxIt, REWARD_MAX_ITER_FRAC, REWARD_WEIGHT_INIT);
    
    % 粒子结构体
    empty_particle.Position = [];
    empty_particle.Velocity = [];
    empty_particle.Cost = [];
    empty_particle.Duration = [];
    empty_particle.Reward = 0;
    empty_particle.Best.Position = [];
    empty_particle.Best.Cost = [];
    empty_particle.Best.Duration = [];
    empty_particle.Best.Reward = 0;
    
    % 初始化群体
    particle = repmat(empty_particle, nPop, 1);
    GlobalBest.Cost = inf;
    GlobalBest.Duration = -inf;
    GlobalBest.Reward = -inf;
    GlobalBest.Position = [];
    
    % 导弹初始速度方向（用于奖赏计算）
    pos_M1     = [20000, 0, 2000];      % 导弹起始位置
    pos_target = [0, 200, 0];           % 真实目标位置
    pos_FY1    = [17800, 0, 1800];      % FY1初始位置
    
    for i = 1:nPop
        % 初始化位置
        particle(i).Position = VarMin + rand(1, nVar) .* (VarMax - VarMin);
        
        % 方位角：使用冯米塞斯分布指向真实目标
        target_direction = pos_target - pos_FY1;
        target_theta = atan2(target_direction(2), target_direction(1));
        particle(i).Position(2) = randVonMises(target_theta, VM_kappa, 1);
        
        % 修复投弹时间约束
        particle(i).Position = fix_time_constraints(particle(i).Position, MIN_INTERVAL, T_SUM_MAX);
        
        particle(i).Velocity = zeros(1, nVar);
        
        % 计算初始目标函数值
        [J, dur, reward] = obj(particle(i).Position, 1);
        particle(i).Cost = J;
        particle(i).Duration = dur;
        particle(i).Reward = reward;
        
        % 个体最优初始化
        particle(i).Best.Position = particle(i).Position;
        particle(i).Best.Cost = J;
        particle(i).Best.Duration = dur;
        particle(i).Best.Reward = reward;
        
        % 更新全局最优
        if particle(i).Best.Cost < GlobalBest.Cost
            GlobalBest = particle(i).Best;
        end
    end
    
    % 记录收敛过程
    BestDuration = nan(MaxIt, 1);
    
    % 停滞检测与多样性注入参数
    stallThreshold = 12;
    improveTol = 1e-6;
    stallCounter = 0;
    prevBestCost = inf;
    
    % 迭代优化
    actual_iterations = MaxIt;  % 默认为最大迭代次数
    for it = 1:MaxIt
        for i = 1:nPop
            % 更新速度
            r1 = rand(1, nVar);
            r2 = rand(1, nVar);
            particle(i).Velocity = w*particle(i).Velocity ...
                + c1*r1.*(particle(i).Best.Position - particle(i).Position) ...
                + c2*r2.*(GlobalBest.Position - particle(i).Position);
            
            % 限幅速度
            particle(i).Velocity = max(particle(i).Velocity, VelMin);
            particle(i).Velocity = min(particle(i).Velocity, VelMax);
            
            % 更新位置
            particle(i).Position = particle(i).Position + particle(i).Velocity;
            
            % 位置越界处理
            particle(i).Position = min(max(particle(i).Position, VarMin), VarMax);
            particle(i).Position(2) = wrapToPi(particle(i).Position(2));  % theta wrap
            
            % 修复投弹时间约束
            particle(i).Position = fix_time_constraints(particle(i).Position, MIN_INTERVAL, T_SUM_MAX);
            
            % 重新计算适应度
            [J, dur, reward] = obj(particle(i).Position, it);
            particle(i).Cost = J;
            particle(i).Duration = dur;
            particle(i).Reward = reward;
            
            % 更新个体最优
            if J < particle(i).Best.Cost
                particle(i).Best.Position = particle(i).Position;
                particle(i).Best.Cost = J;
                particle(i).Best.Duration = dur;
                particle(i).Best.Reward = reward;
            end
            
            % 更新全局最优
            if it <= REWARD_MAX_ITER_FRAC * MaxIt
                % 前期迭代：考虑Cost和Reward的综合效果
                if particle(i).Best.Cost < GlobalBest.Cost || ...
                   (abs(particle(i).Best.Cost - GlobalBest.Cost) < 1e-6 && ...
                    particle(i).Best.Reward > GlobalBest.Reward)
                    GlobalBest = particle(i).Best;
                end
            else
                % 后期迭代：仅考虑Cost（遮挡时长）
                if particle(i).Best.Cost < GlobalBest.Cost
                    GlobalBest = particle(i).Best;
                end
            end
        end
        
        BestDuration(it) = GlobalBest.Duration;
        
        % 停滞检测：统计BestCost是否改进
        if it == 1 || (GlobalBest.Cost < prevBestCost - improveTol)
            prevBestCost = GlobalBest.Cost;
            stallCounter = 0;
        else
            stallCounter = stallCounter + 1;
        end
        
        % 提前停止判断
        if stallCounter >= stallThreshold
            actual_iterations = it;
            break;
        end
        
        % 衰减惯性权重
        w = w * wDamp;
    end
    
    % 返回结果
    best_position = GlobalBest.Position;
    best_duration = GlobalBest.Duration;
    best_reward = GlobalBest.Reward;
    convergence_curve = BestDuration(1:actual_iterations);
end

%% 辅助函数：修复时间约束
function pos = fix_time_constraints(pos, min_interval, t_sum_max)
    % 1. 确保时间顺序和最小间隔
    t1 = pos(3); dt1 = pos(4);
    t2 = pos(5); dt2 = pos(6);
    t3 = pos(7); dt3 = pos(8);
    
    % 按照大小排序三个投弹时间
    times = [t1, t2, t3];
    [sorted_times, ~] = sort(times);
    
    % 应用最小间隔要求
    sorted_times(2) = max(sorted_times(2), sorted_times(1) + min_interval);
    sorted_times(3) = max(sorted_times(3), sorted_times(2) + min_interval);
    
    % 将排序后的时间按升序赋回
    t1 = sorted_times(1);
    t2 = sorted_times(2);
    t3 = sorted_times(3);
    
    % 2. 确保所有投弹时间小于等于T_SUM_MAX
    if t1 + dt1 > t_sum_max
        dt1 = t_sum_max - t1;
    end
    if t2 + dt2 > t_sum_max
        dt2 = t_sum_max - t2;
    end
    if t3 + dt3 > t_sum_max
        dt3 = t_sum_max - t3;
    end
    
    % 确保dt为正数
    dt1 = max(dt1, 0);
    dt2 = max(dt2, 0);
    dt3 = max(dt3, 0);
    
    % 更新位置
    pos(3) = t1; pos(4) = dt1;
    pos(5) = t2; pos(6) = dt2;
    pos(7) = t3; pos(8) = dt3;
end

%% 目标函数：从 [s,theta,t1,dt1,t2,dt2,t3,dt3] 计算总遮挡（带奖赏机制）
function [J, duration, reward] = objective_function(x, q1_opts, it, MaxIt, REWARD_MAX_ITER_FRAC, REWARD_WEIGHT_INIT)
    % 提取参数
    s = x(1); th = x(2);
    t1 = x(3); dt1 = x(4);
    t2 = x(5); dt2 = x(6);
    t3 = x(7); dt3 = x(8);
    
    % 计算速度分量
    vx = s * cos(th); vy = s * sin(th);
    
    % 计算总遮挡时长
    try
        % 调用计算总遮挡时长的函数
        [duration, ~] = calculate_total_occlusion(vx, vy, [t1, t2, t3], [dt1, dt2, dt3], q1_opts);
    catch
        duration = 0;
    end
    
    % 计算方向奖赏
    % 计算FY1方向与导弹方向的夹角
    pos_M1 = [20000, 0, 2000];  % 导弹起始位置
    pos_target = [0, 200, 0];   % 真实目标位置
    pos_FY1 = [17800, 0, 1800]; % FY1初始位置
    
    % 计算导弹指向真实目标的方向向量
    v_M = 300;
    vv_M1 = v_M * (pos_target - pos_M1) / norm(pos_target - pos_M1);
    vv_FY1 = [vx, vy, 0];  % FY1速度向量
    
    % 计算夹角
    cos_angle = dot(vv_FY1, vv_M1) / (norm(vv_FY1) * norm(vv_M1));
    angle = acos(max(min(cos_angle, 1), -1));  % 防止数值误差
    
    % 使用Weibull分布计算奖赏
    normalized_angle = angle / pi;
    lambda = 1;
    k_shape = 3.5;
    reward = exp(-((normalized_angle/lambda)^k_shape));
    
    % 随迭代逐渐减小奖赏权重
    reward_weight = max(0, REWARD_WEIGHT_INIT * (1 - it / (REWARD_MAX_ITER_FRAC * MaxIt)));
    
    % 最终目标函数
    J = -duration - reward_weight * reward;
end

%% 辅助函数：计算总遮挡时长
function [totalDuration, mergedIntervals] = calculate_total_occlusion(vx, vy, t_throws, t_explodes, opts)
    % 分别计算每次投弹产生的遮挡区间
    numBombs = length(t_throws);
    allIntervals = [];
    
    % 计算每次投弹的遮挡区间
    for i = 1:numBombs
        [dur, intervals] = q1_occlusion_time(vx, vy, t_throws(i), t_explodes(i), opts);
        if ~isempty(intervals)
            offset = t_throws(i) + t_explodes(i);  % 将相对引爆时刻的区间偏移到全局时间轴
            allIntervals = [allIntervals; intervals + offset];
        end
    end
    
    % 合并重叠区间
    if isempty(allIntervals)
        totalDuration = 0;
        mergedIntervals = [];
    else
        % 按开始时间排序
        allIntervals = sortrows(allIntervals, 1);
        merged = allIntervals(1, :);
        
        % 合并重叠区间
        for i = 2:size(allIntervals, 1)
            last = merged(end, :);
            curr = allIntervals(i, :);
            
            if curr(1) <= last(2)
                % 区间重叠，合并
                merged(end, 2) = max(last(2), curr(2));
            else
                % 无重叠，添加新区间
                merged = [merged; curr];
            end
        end
        
        % 计算总时长
        totalDuration = sum(merged(:, 2) - merged(:, 1));
        mergedIntervals = merged;
    end
end

%% von Mises 随机采样
function theta = randVonMises(mu, kappa, n)
    if kappa < 1e-8
        theta = (rand(n,1)*2*pi - pi) + mu; % 近似均匀
        theta = mod(theta + pi, 2*pi) - pi;
        return;
    end
    a = 1 + sqrt(1 + 4*kappa^2);
    b = (a - sqrt(2*a)) / (2*kappa);
    r = (1 + b^2) / (2*b);
    theta = zeros(n,1);
    for ii = 1:n
        while true
            u1 = rand; u2 = rand; u3 = rand;
            z = cos(pi*u1);
            f = (1 + r*z) / (r + z);
            c = kappa * (r - f);
            if (u2 < c*(2 - c)) || (log(u2) <= (c - 1))
                break;
            end
        end
        signv = 1;
        if u3 - 0.5 < 0, signv = -1; end
        theta(ii) = mu + signv * acos(f);
    end
    theta = mod(theta + pi, 2*pi) - pi; % wrap到[-pi,pi]
end 