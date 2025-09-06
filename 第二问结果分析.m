% 第二问结果分析：执行100次PSO优化并统计结果
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
all_best_positions = zeros(num_runs, 4);  % 最佳参数 [s, theta, t_throw, t_explode]
all_iterations = zeros(num_runs, 1);      % 收敛所需迭代次数
all_convergence = cell(num_runs, 1);      % 收敛曲线数据

% 优化参数（与Question2.m保持一致）
max_iterations = 120;  % 最大迭代次数

%% 并行执行100次优化
disp('开始并行执行100次优化...');
tic;  % 开始计时

parfor run_idx = 1:num_runs
    % 调用优化函数
    [best_position, best_duration, convergence, actual_iterations] = run_pso_optimization();
    
    % 存储结果
    all_best_durations(run_idx) = best_duration;
    all_best_positions(run_idx, :) = best_position;
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
t0_best = best_of_all(3);
t1_best = best_of_all(4);
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
results.global_best.vx = vx_best;
results.global_best.vy = vy_best;
results.mean_convergence = mean_convergence;
results.runtime = runtime;

% 生成带时间戳的文件名
timestamp = datestr(now, 'yyyymmdd_HHMMSS');
filename = sprintf('PSO_Q2_Results_%s.mat', timestamp);

% 保存结果
save(filename, 'results');
fprintf('结果已保存到文件: %s\n', filename);

%% 结果输出
fprintf('\n========== 结果统计 ==========\n');
fprintf('最佳遮挡时长: %.6f 秒\n', max_duration);
fprintf('平均遮挡时长: %.6f 秒 (std: %.6f)\n', mean_duration, std_duration);
fprintf('最短遮挡时长: %.6f 秒\n', min_duration);
fprintf('中位数遮挡时长: %.6f 秒\n', median_duration);
fprintf('四分位数 Q1/Q3: %.6f / %.6f 秒\n', q1_duration, q3_duration);
fprintf('平均收敛迭代数: %.2f\n', mean(all_iterations));

fprintf('\n========== 全局最优解 ==========\n');
fprintf('  s (speed) = %.6f m/s\n', s_best);
fprintf('  theta (rad) = %.6f (%.2f度)\n', th_best, rad2deg(th_best));
fprintf('  vx_FY1 = %.6f m/s\n', vx_best);
fprintf('  vy_FY1 = %.6f m/s\n', vy_best);
fprintf('  t_throw = %.6f s\n', t0_best);
fprintf('  t_explode = %.6f s\n', t1_best);

%% 可视化结果
% 保存图表设置
save_figures = true;  % 是否保存图表到文件
fig_format = 'png';   % 图表保存格式（'png', 'jpg', 'fig'等）
fig_folder = 'PSO_Q2_Figures';  % 图表保存文件夹

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
    fig_file = fullfile(fig_folder, sprintf('PSO_Q2_收敛曲线_%s.%s', timestamp, fig_format));
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
    fig_file = fullfile(fig_folder, sprintf('PSO_Q2_遮挡时长分布_%s.%s', timestamp, fig_format));
    saveas(gcf, fig_file);
    fprintf('遮挡时长分布图已保存: %s\n', fig_file);
end

% 3. 参数分布散点图
figure('Name', '参数散点图', 'Position', [900, 100, 1000, 800]);

% s 和 theta 的散点图
subplot(2,2,1);
scatter(all_best_positions(:,1), all_best_positions(:,2), 50, all_best_durations, 'filled');
colorbar; grid on;
xlabel('速度大小 s (m/s)'); ylabel('方位角 theta (rad)');
title('速度参数与遮挡时长关系');
colormap(jet);

% t_throw 和 t_explode 的散点图
subplot(2,2,2);
scatter(all_best_positions(:,3), all_best_positions(:,4), 50, all_best_durations, 'filled');
colorbar; grid on;
xlabel('投弹时刻 t\_throw (s)'); ylabel('引爆时间 t\_explode (s)');
title('时间参数与遮挡时长关系');

% vx 和 vy 的散点图
vx = all_best_positions(:,1) .* cos(all_best_positions(:,2));
vy = all_best_positions(:,1) .* sin(all_best_positions(:,2));
subplot(2,2,3);
scatter(vx, vy, 50, all_best_durations, 'filled');
colorbar; grid on;
xlabel('水平速度 vx (m/s)'); ylabel('水平速度 vy (m/s)');
title('速度分量与遮挡时长关系');

% 遮挡时长与迭代次数的关系
subplot(2,2,4);
scatter(all_iterations, all_best_durations, 50, 'filled');
grid on;
xlabel('收敛迭代次数'); ylabel('遮挡时长 (s)');
title('收敛速度与优化效果关系');

% 保存图表
if save_figures
    fig_file = fullfile(fig_folder, sprintf('PSO_Q2_参数散点图_%s.%s', timestamp, fig_format));
    saveas(gcf, fig_file);
    fprintf('参数散点图已保存: %s\n', fig_file);
end

% 额外保存一份总结报告
if save_figures
    % 创建文本文件保存统计结果
    report_file = fullfile(fig_folder, sprintf('PSO_Q2_统计报告_%s.txt', timestamp));
    fid = fopen(report_file, 'w');
    if fid ~= -1
        fprintf(fid, '==========================================\n');
        fprintf(fid, '  第二问PSO优化结果统计报告 (%s)\n', timestamp);
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
        fprintf(fid, '投弹时刻 t_throw = %.6f s\n', t0_best);
        fprintf(fid, '引爆时间 t_explode = %.6f s\n\n', t1_best);
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
function [best_position, best_duration, convergence_curve, actual_iterations] = run_pso_optimization()
    % 约束与场景参数（与Question2.m保持一致）
    SPEED_MIN = 70; SPEED_MAX = 140; 
    THETA_MIN = -pi; THETA_MAX = pi; 
    VM_kappa = 12;
    T_MAX = 20;
    T_SUM_MAX = 20 * (1 - 1e-6);
    q1_opts = struct('tSimEnd', 20, 'g', 9.8);
    
    % 粒子群参数
    nVar = 4;
    VarMin = [SPEED_MIN, THETA_MIN, 0.0, 0.0];
    VarMax = [SPEED_MAX, THETA_MAX, T_MAX, T_MAX];
    
    nPop = 100;
    MaxIt = 120;
    w = 0.72;
    wDamp = 0.99;
    c1 = 2.0;
    c2 = 1.6;
    
    VelMax = 0.25 * (VarMax - VarMin);
    VelMin = -VelMax;
    
    % 目标函数（最小化：-遮挡时长）
    obj = @(x) objective_s_theta(x, q1_opts);
    
    % 粒子结构体
    empty_particle.Position = [];
    empty_particle.Velocity = [];
    empty_particle.Cost = [];
    empty_particle.Duration = [];
    empty_particle.Best.Position = [];
    empty_particle.Best.Cost = [];
    empty_particle.Best.Duration = [];
    
    % 初始化群体
    particle = repmat(empty_particle, nPop, 1);
    GlobalBest.Cost = inf;
    GlobalBest.Duration = -inf;
    GlobalBest.Position = [];
    
    for i = 1:nPop
        particle(i).Position = VarMin + rand(1, nVar) .* (VarMax - VarMin);
        % 速度大小均匀
        particle(i).Position(1) = SPEED_MIN + rand*(SPEED_MAX - SPEED_MIN);
        % 方位角
        if rand < 0.5
            mu_sample = 0;
        else
            mu_sample = pi;
        end
        particle(i).Position(2) = randVonMises(mu_sample, VM_kappa, 1);
        % 强可行修复：保证 t0+t1 <= T_SUM_MAX
        t0i = particle(i).Position(3); t1i = particle(i).Position(4);
        tsum = t0i + t1i;
        if tsum > T_SUM_MAX
            sf = T_SUM_MAX / tsum; particle(i).Position(3:4) = particle(i).Position(3:4) * sf;
        end
        particle(i).Velocity = zeros(1, nVar);

        [J, dur] = obj(particle(i).Position);
        particle(i).Cost = J;
        particle(i).Duration = dur;

        particle(i).Best.Position = particle(i).Position;
        particle(i).Best.Cost = J;
        particle(i).Best.Duration = dur;

        if particle(i).Best.Cost < GlobalBest.Cost
            GlobalBest = particle(i).Best;
        end
    end
    
    BestDuration = nan(MaxIt, 1);
    
    % 提前收敛检测变量
    no_improve_count = 0;
    convergence_threshold = 10; % 10次迭代无改进则判定收敛
    min_improvement = 1e-6;  % 最小改进阈值
    last_best_cost = GlobalBest.Cost;
    
    % 迭代
    actual_iterations = MaxIt; % 默认最大迭代次数
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
            particle(i).Position(2) = wrapToPi(particle(i).Position(2)); % theta wrap
            % 强可行修复
            t0i = particle(i).Position(3); t1i = particle(i).Position(4);
            tsum = t0i + t1i;
            if tsum > T_SUM_MAX
                sf = T_SUM_MAX / tsum; particle(i).Position(3:4) = particle(i).Position(3:4) * sf;
            end

            % 重新计算适应度
            [J, dur] = obj(particle(i).Position);
            particle(i).Cost = J;
            particle(i).Duration = dur;

            % 更新个体最优
            if J < particle(i).Best.Cost
                particle(i).Best.Position = particle(i).Position;
                particle(i).Best.Cost = J;
                particle(i).Best.Duration = dur;
            end

            % 更新全局最优
            if particle(i).Best.Cost < GlobalBest.Cost
                GlobalBest = particle(i).Best;
            end
        end

        BestDuration(it) = GlobalBest.Duration;
        
        % 衰减惯性权重
        w = w * wDamp;
        
        % 收敛检测
        if abs(GlobalBest.Cost - last_best_cost) < min_improvement
            no_improve_count = no_improve_count + 1;
            if no_improve_count >= convergence_threshold
                actual_iterations = it;
                break; % 提前结束迭代
            end
        else
            no_improve_count = 0;  % 重置计数器
            last_best_cost = GlobalBest.Cost;
        end
    end
    
    % 返回结果
    best_position = GlobalBest.Position;
    best_duration = GlobalBest.Duration;
    convergence_curve = BestDuration(1:actual_iterations);
end

% 目标函数
function [J, duration] = objective_s_theta(x, q1_opts)
    s = x(1); th = x(2); t0 = x(3); t1 = x(4);
    vx = s * cos(th); vy = s * sin(th);
    try
        [duration, ~] = q1_occlusion_time(vx, vy, t0, t1, q1_opts);
    catch
        duration = 0;
    end
    J = -duration;
end

% von Mises 随机采样
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