% Question 4 Results Analysis: Run PSO optimization 100 times in parallel (20 cores)
% 说明：
% - 并行开启20核（若可用），并行执行100次第四问的最优化（Question4的三架无人机各投一弹）
% - 复用第四问的PSO与目标函数逻辑（在本脚本中实现独立的 run_pso_optimization_q4）
% - 汇总每次最佳结果，保存MAT文件与报告，并进行可视化

clear; clc;

%% 配置并行计算池（目标20核）
target_cores = 20;
try
    c = parcluster('local');
    fprintf('当前默认最大工作进程数: %d\n', c.NumWorkers);
    c.NumWorkers = target_cores;
    saveProfile(c);

    poolobj = gcp('nocreate');
    if isempty(poolobj)
        parpool(c, target_cores);
        poolobj = gcp;
        fprintf('成功创建并行计算池，使用 %d 个核心\n', poolobj.NumWorkers);
    else
        fprintf('已存在并行池，当前使用 %d 个核心\n', poolobj.NumWorkers);
        if poolobj.NumWorkers ~= target_cores
            fprintf('删除现有并行池以调整核心数...\n');
            delete(poolobj);
            parpool(c, target_cores);
            poolobj = gcp;
            fprintf('调整并行计算池至 %d 个核心\n', poolobj.NumWorkers);
        end
    end
    fprintf('实际使用的核心数: %d\n', poolobj.NumWorkers);
catch err
    warning('创建并行池时出错: %s', err.message);
    disp('将尝试使用最大可用核心数');
    try
        parpool('local');
        poolobj = gcp;
        fprintf('使用默认配置创建并行池，核心数: %d\n', poolobj.NumWorkers);
    catch
        warning('无法创建并行池，将使用串行计算');
    end
end

%% 参数：重复运行次数
num_runs = 100;

%% 结果容器
all_best_durations   = zeros(num_runs, 1);     % 最佳遮挡时长
all_best_positions   = zeros(num_runs, 12);    % 最佳参数 [s1 th1 t1 dt1  s2 th2 t2 dt2  s3 th3 t3 dt3]
all_best_rewards     = zeros(num_runs, 1);     % 最佳奖赏值
all_iterations       = zeros(num_runs, 1);     % 实际迭代次数（如有提前停止则小于MaxIt）
all_convergence      = cell(num_runs, 1);      % 每次的收敛曲线（BestDuration）
all_cloud_durations  = zeros(num_runs, 3);     % 三个云团的遮挡贡献（针对最佳解）

fprintf('开始并行执行 %d 次第四问优化...\n', num_runs);
tStart = tic;

parfor run_idx = 1:num_runs
    [best_position, best_duration, best_reward, cloud_durs, convergence_curve, actual_iterations] = run_pso_optimization_q4();

    all_best_durations(run_idx)  = best_duration;
    all_best_positions(run_idx,:)= best_position;
    all_best_rewards(run_idx)    = best_reward;
    all_iterations(run_idx)      = actual_iterations;
    all_convergence{run_idx}     = convergence_curve;
    if ~isempty(cloud_durs)
        all_cloud_durations(run_idx, :) = cloud_durs(:)';
    end

    if mod(run_idx, 10) == 0
        fprintf('完成第 %d/%d 次优化\n', run_idx, num_runs);
    end
end

runtime = toc(tStart);
fprintf('完成 %d 次优化，总耗时: %.2f 秒\n', num_runs, runtime);

%% 统计与汇总
[max_duration, max_idx] = max(all_best_durations);
best_of_all = all_best_positions(max_idx, :);

s1_best  = best_of_all(1);  th1_best = best_of_all(2);  t1_best  = best_of_all(3);  dt1_best = best_of_all(4);
s2_best  = best_of_all(5);  th2_best = best_of_all(6);  t2_best  = best_of_all(7);  dt2_best = best_of_all(8);
s3_best  = best_of_all(9);  th3_best = best_of_all(10); t3_best  = best_of_all(11); dt3_best = best_of_all(12);

best_reward     = all_best_rewards(max_idx);
best_cloud_durs = all_cloud_durations(max_idx, :);

mean_duration = mean(all_best_durations);
std_duration  = std(all_best_durations);
median_duration = median(all_best_durations);
min_duration  = min(all_best_durations);
q1_duration   = quantile(all_best_durations, 0.25);
q3_duration   = quantile(all_best_durations, 0.75);

% 平均收敛曲线（长度对齐）
max_len = max(cellfun(@length, all_convergence));
padded_curves = zeros(num_runs, max_len);
for i = 1:num_runs
    curve = all_convergence{i};
    if isempty(curve)
        continue;
    end
    padded_curves(i, 1:length(curve)) = curve(:)';
    if length(curve) < max_len
        padded_curves(i, length(curve)+1:end) = curve(end);
    end
end
mean_convergence = mean(padded_curves, 1);

%% 保存结果到 MAT 文件
results = struct();
results.best_durations   = all_best_durations;
results.best_positions   = all_best_positions;
results.best_rewards     = all_best_rewards;
results.iterations       = all_iterations;
results.convergence      = all_convergence;
results.cloud_durations  = all_cloud_durations;
results.statistics.max_duration   = max_duration;
results.statistics.mean_duration  = mean_duration;
results.statistics.std_duration   = std_duration;
results.statistics.median_duration= median_duration;
results.statistics.q1_duration    = q1_duration;
results.statistics.q3_duration    = q3_duration;
results.global_best.position = best_of_all;
results.global_best.duration = max_duration;
results.global_best.reward   = best_reward;
results.global_best.cloud_durations = best_cloud_durs;
results.mean_convergence = mean_convergence;
results.runtime = runtime;

timestamp = datestr(now, 'yyyymmdd_HHMMSS');
mat_filename = sprintf('PSO_Q4_Results_%s.mat', timestamp);
save(mat_filename, 'results');
fprintf('结果已保存到文件: %s\n', mat_filename);

%% 可视化与保存图表
save_figures = true;
fig_format   = 'png';
fig_folder   = 'PSO_Q4_Figures';
if save_figures
    if ~exist(fig_folder, 'dir')
        mkdir(fig_folder);
        fprintf('创建图表保存文件夹: %s\n', fig_folder);
    end
end

% 1. 平均收敛曲线
figure('Name', 'Q4 平均收敛曲线', 'Position', [100, 100, 900, 450]);
plot(1:length(mean_convergence), -mean_convergence, 'LineWidth', 2); grid on; box on;
xlabel('迭代次数'); ylabel('平均最佳遮挡时长 (秒)'); title('第四问：100次PSO的平均收敛曲线');
if save_figures
    f1 = fullfile(fig_folder, sprintf('Q4_收敛曲线_%s.%s', timestamp, fig_format));
    saveas(gcf, f1); fprintf('收敛曲线已保存: %s\n', f1);
end

% 2. 遮挡时长分布
figure('Name', 'Q4 遮挡时长分布', 'Position', [100, 600, 900, 450]);
histogram(all_best_durations, 20, 'Normalization', 'count'); hold on;
line([mean_duration, mean_duration], ylim, 'Color', 'r', 'LineWidth', 2, 'LineStyle', '--');
line([max_duration, max_duration], ylim, 'Color', 'g', 'LineWidth', 2, 'LineStyle', '--');
legend('频数', '平均', '最大'); grid on; box on;
xlabel('最佳遮挡时长 (秒)'); ylabel('频数'); title('第四问：遮挡时长分布');
if save_figures
    f2 = fullfile(fig_folder, sprintf('Q4_遮挡时长分布_%s.%s', timestamp, fig_format));
    saveas(gcf, f2); fprintf('遮挡时长分布已保存: %s\n', f2);
end

% 3. 参数散点图（速度-方位角; 投弹时刻; 引爆延迟）
figure('Name', 'Q4 参数散点图', 'Position', [1000, 100, 1100, 850]);
colormap(jet);

subplot(2,2,1);
scatter(all_best_positions(:,1), all_best_positions(:,2), 40, all_best_durations, 'filled');
colorbar; grid on; xlabel('s1 (m/s)'); ylabel('th1 (rad)'); title('FY1: s-theta 与遮挡');

subplot(2,2,2);
scatter(all_best_positions(:,5), all_best_positions(:,6), 40, all_best_durations, 'filled');
colorbar; grid on; xlabel('s2 (m/s)'); ylabel('th2 (rad)'); title('FY2: s-theta 与遮挡');

subplot(2,2,3);
scatter(all_best_positions(:,9), all_best_positions(:,10), 40, all_best_durations, 'filled');
colorbar; grid on; xlabel('s3 (m/s)'); ylabel('th3 (rad)'); title('FY3: s-theta 与遮挡');

subplot(2,2,4);
scatter3(all_best_positions(:,3), all_best_positions(:,7), all_best_positions(:,11), 36, all_best_durations, 'filled');
colorbar; grid on; xlabel('t1 (s)'); ylabel('t2 (s)'); zlabel('t3 (s)');
title('三次投弹时刻与遮挡'); view(45,30);

if save_figures
    f3 = fullfile(fig_folder, sprintf('Q4_参数散点图_%s.%s', timestamp, fig_format));
    saveas(gcf, f3); fprintf('参数散点图已保存: %s\n', f3);
end

% 4. 引爆延迟的三维散点
figure('Name', 'Q4 引爆延迟散点', 'Position', [100, 100, 900, 450]);
scatter3(all_best_positions(:,4), all_best_positions(:,8), all_best_positions(:,12), 36, all_best_durations, 'filled');
colorbar; grid on; xlabel('dt1 (s)'); ylabel('dt2 (s)'); zlabel('dt3 (s)');
title('三次引爆延迟与遮挡'); view(45,30);
if save_figures
    f4 = fullfile(fig_folder, sprintf('Q4_延迟散点_%s.%s', timestamp, fig_format));
    saveas(gcf, f4); fprintf('延迟散点图已保存: %s\n', f4);
end

% 5. 最优解的三云团遮挡贡献条形图
figure('Name', 'Q4 云团贡献(最优)', 'Position', [1050, 500, 600, 380]);
bar(categorical({'Cloud1','Cloud2','Cloud3'}), best_cloud_durs);
ylabel('遮挡时长 (秒)'); title('最优解三云团遮挡贡献'); grid on;
if save_figures
    f5 = fullfile(fig_folder, sprintf('Q4_云团贡献_%s.%s', timestamp, fig_format));
    saveas(gcf, f5); fprintf('云团贡献图已保存: %s\n', f5);
end

%% 输出摘要
fprintf('\n========== 第四问 PSO 优化结果统计 ==========''\n');
fprintf('\n最佳遮挡时长: %.6f 秒\n', max_duration);
fprintf('平均遮挡时长: %.6f 秒 (std: %.6f)\n', mean_duration, std_duration);
fprintf('最短遮挡时长: %.6f 秒\n', min_duration);
fprintf('中位数遮挡时长: %.6f 秒\n', median_duration);
fprintf('四分位数 Q1/Q3: %.6f / %.6f 秒\n', q1_duration, q3_duration);

fprintf('\n========== 全局最优解（一次运行中的最佳） ==========''\n');
fprintf('FY1: s=%.6f, th=%.6f, t=%.6f, dt=%.6f\n', s1_best, th1_best, t1_best, dt1_best);
fprintf('FY2: s=%.6f, th=%.6f, t=%.6f, dt=%.6f\n', s2_best, th2_best, t2_best, dt2_best);
fprintf('FY3: s=%.6f, th=%.6f, t=%.6f, dt=%.6f\n', s3_best, th3_best, t3_best, dt3_best);
fprintf('最佳奖赏值: %.6f\n', best_reward);
fprintf('云团贡献: [%.2f, %.2f, %.2f] 秒\n', best_cloud_durs(1), best_cloud_durs(2), best_cloud_durs(3));

%% 额外保存文本报告
report_file = fullfile(fig_folder, sprintf('PSO_Q4_统计报告_%s.txt', timestamp));
fid = fopen(report_file, 'w');
if fid ~= -1
    fprintf(fid, '==========================================\n');
    fprintf(fid, '  第四问 PSO 优化结果统计报告 (%s)\n', timestamp);
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
    fprintf(fid, 'FY1: s=%.6f, th=%.6f, t=%.6f, dt=%.6f\n', s1_best, th1_best, t1_best, dt1_best);
    fprintf(fid, 'FY2: s=%.6f, th=%.6f, t=%.6f, dt=%.6f\n', s2_best, th2_best, t2_best, dt2_best);
    fprintf(fid, 'FY3: s=%.6f, th=%.6f, t=%.6f, dt=%.6f\n', s3_best, th3_best, t3_best, dt3_best);
    fprintf(fid, '最佳奖赏值: %.6f\n', best_reward);
    fprintf(fid, '云团贡献: [%.2f, %.2f, %.2f] 秒\n\n', best_cloud_durs(1), best_cloud_durs(2), best_cloud_durs(3));

    fprintf(fid, '结果数据文件: %s\n', mat_filename);
    fclose(fid);
    fprintf('统计报告已保存: %s\n', report_file);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 子函数：执行一次第四问的PSO优化，返回最佳解与收敛信息
function [best_position, best_duration, best_reward, cloud_durations, convergence_curve, actual_iterations] = run_pso_optimization_q4()
    % 约束与场景常量（与 Question4.m 保持一致）
    SPEED_MIN = 70; SPEED_MAX = 140;     % s ∈ [70,140]
    THETA_MIN = -pi; THETA_MAX = pi;     % theta ∈ [-pi, pi]
    VM_kappa = 10;                       % 初始方位角冯米塞斯浓度

    % 时间约束
    T_MAX = 67;                           %#ok<NASGU>

    % 目标函数仿真选项
    sim_opts = struct('tSimEnd', 20, 'g', 9.8);

    % 目标/导弹/云团参数
    v_cloud   = 3;                        % 云团垂直坠落速度
    r_cloud   = 10;                       % 云团半径
    r_target  = 7;                        % 目标半径
    h_target  = 10;                       % 目标高度
    pos_target = [0, 200, 0];
    pos_M1     = [20000, 0, 2000];
    v_M       = 300;                      % 导弹速度
    vv_M1     = v_M * (pos_target - pos_M1) / norm(pos_target - pos_M1);

    % 三架无人机初始位置
    pos_FY1    = [17800, 0, 1800];
    pos_FY2    = [12000, 1400, 1400];
    pos_FY3    = [6000,  -3000, 700];

    % 奖赏机制参数（退火）
    REWARD_MAX_ITER_FRAC = 0.5;
    REWARD_WEIGHT_INIT   = 1;
    DIST_SCALE           = 300;           % 距离尺度（m）

    % 粒子群参数与搜索边界
    nVar  = 12; % [s1 th1 t1 dt1  s2 th2 t2 dt2  s3 th3 t3 dt3]
    VarMin = [SPEED_MIN, THETA_MIN, 0, 0,  SPEED_MIN, THETA_MIN, 0, 0,  SPEED_MIN, THETA_MIN, 0, 0];
    VarMax = [SPEED_MAX, THETA_MAX, T_MAX, T_MAX,  SPEED_MAX, THETA_MAX, T_MAX, T_MAX,  SPEED_MAX, THETA_MAX, T_MAX, T_MAX];

    nPop = 150;              % 群体规模
    MaxIt = 150;             % 最大迭代数
    w = 1.2;                 % 惯性权重
    wDamp = 0.99;            % 惯性权重衰减
    c1 = 1.6;                % 个体学习因子
    c2 = 2.0;                % 群体学习因子

    % 速度上限
    VelMax = 0.25 * (VarMax - VarMin);
    VelMin = -VelMax;

    % 目标函数（最小化：-遮挡时长 - 奖励项）
    obj = @(x, it) objective_q4(x, sim_opts, ...
        v_cloud, r_cloud, r_target, h_target, [0, 200, h_target/2], pos_M1, vv_M1, ...
        pos_FY1, pos_FY2, pos_FY3, ...
        REWARD_MAX_ITER_FRAC, REWARD_WEIGHT_INIT, DIST_SCALE, it, MaxIt);

    % 粒子结构体
    empty_particle.Position = [];
    empty_particle.Velocity = [];
    empty_particle.Cost     = [];
    empty_particle.Duration = [];
    empty_particle.Reward   = 0;
    empty_particle.CloudDurations = [];
    empty_particle.Best = empty_particle;

    particle  = repmat(empty_particle, nPop, 1);
    GlobalBest.Cost = inf; GlobalBest.Duration = -inf; GlobalBest.Reward = -inf; GlobalBest.Position = [];
    GlobalBest.CloudDurations = [];

    % 初始化群体
    for i = 1:nPop
        particle(i).Position = VarMin + rand(1, nVar) .* (VarMax - VarMin);

        % FY1的方位角：使用冯米塞斯分布指向真实目标（0或pi的混合）
        if rand < 0.5
            mu_sample = 0;
        else
            mu_sample = pi;
        end
        particle(i).Position(2) = randVonMises(mu_sample, VM_kappa, 1);

        % FY2的方位角：指向导弹和目标连线中点，k=2.5
        midpoint = (pos_M1 + pos_target) / 2;
        midpoint_direction = midpoint - pos_FY2;
        midpoint_theta = atan2(midpoint_direction(2), midpoint_direction(1));
        particle(i).Position(6) = randVonMises(midpoint_theta, 2.5, 1);

        % FY3的方位角：单边冯米塞斯分布，k=1.5
        target_direction_FY3 = pos_target - pos_FY3;
        target_theta_FY3 = atan2(target_direction_FY3(2), target_direction_FY3(1));
        theta_raw = randVonMises(0, 1.5, 1);   % 中心在0
        theta_abs = abs(theta_raw);            % 单边
        particle(i).Position(10) = wrapToPi(target_theta_FY3 + theta_abs);

        particle(i).Velocity = zeros(1, nVar);

        [J, dur, rew, cloud_durs] = obj(particle(i).Position, 1);
        particle(i).Cost = J;
        particle(i).Duration = dur;
        particle(i).Reward = rew;
        particle(i).CloudDurations = cloud_durs;

        particle(i).Best.Position = particle(i).Position;
        particle(i).Best.Cost = J;
        particle(i).Best.Duration = dur;
        particle(i).Best.Reward = rew;
        particle(i).Best.CloudDurations = cloud_durs;

        if particle(i).Best.Cost < GlobalBest.Cost
            GlobalBest = particle(i).Best;
        end
    end

    % 记录收敛
    BestDuration = nan(MaxIt,1);

    % 停滞检测（多样性注入）
    stallThreshold = 18; improveTol = 1e-6; stallCounter = 0; prevBestCost = inf;

    % 迭代优化
    actual_iterations = MaxIt;
    for it = 1:MaxIt
        for i = 1:nPop
            r1 = rand(1, nVar); r2 = rand(1, nVar);
            particle(i).Velocity = w*particle(i).Velocity ...
                + c1*r1.*(particle(i).Best.Position - particle(i).Position) ...
                + c2*r2.*(GlobalBest.Position - particle(i).Position);
            particle(i).Velocity = min(max(particle(i).Velocity, VelMin), VelMax);
            particle(i).Position = particle(i).Position + particle(i).Velocity;
            particle(i).Position = min(max(particle(i).Position, VarMin), VarMax);
            particle(i).Position([2,6,10]) = arrayfun(@wrapToPi, particle(i).Position([2,6,10]));

            [J, dur, rew, cloud_durs] = obj(particle(i).Position, it);
            particle(i).Cost = J; particle(i).Duration = dur; particle(i).Reward = rew;
            particle(i).CloudDurations = cloud_durs;

            if J < particle(i).Best.Cost
                particle(i).Best = particle(i);
            end

            if it <= REWARD_MAX_ITER_FRAC * MaxIt
                if particle(i).Best.Cost < GlobalBest.Cost || ...
                   (abs(particle(i).Best.Cost - GlobalBest.Cost) < 1e-6 && ...
                    particle(i).Best.Reward > GlobalBest.Reward)
                    GlobalBest = particle(i).Best;
                end
            else
                if particle(i).Best.Cost < GlobalBest.Cost
                    GlobalBest = particle(i).Best;
                end
            end
        end

        BestDuration(it) = GlobalBest.Duration;

        % 停滞检测与多样性注入
        if it == 1 || (GlobalBest.Cost < prevBestCost - improveTol)
            prevBestCost = GlobalBest.Cost; stallCounter = 0;
        else
            stallCounter = stallCounter + 1;
        end
        if stallCounter >= stallThreshold
            numReseed = max(1, round(0.10 * nPop));
            costs = [particle.Cost]; [~, idxSorted] = sort(costs, 'descend'); reseedIdx = idxSorted(1:numReseed);
            for k = 1:numReseed
                j = reseedIdx(k); pos = particle(j).Position;
                % 时间维（t,dt）高斯扰动（标准差按各维度范围的12%）
                timeIdx = [3,4, 7,8, 11,12];
                timeRange = VarMax(timeIdx) - VarMin(timeIdx);
                timeStd = 0.12 * timeRange;
                pos(timeIdx) = pos(timeIdx) + randn(1,6).*timeStd;
                pos(timeIdx) = VarMin(timeIdx) + rand(1,6).*(VarMax(timeIdx)-VarMin(timeIdx));
                % s/theta 高斯扰动
                thetaStd = deg2rad(20);
                pos([2,6,10]) = arrayfun(@(th) wrapToPi(th + randn*thetaStd), pos([2,6,10]));
                sStd = 0.08*(SPEED_MAX-SPEED_MIN);
                pos([1,5,9]) = min(max(pos([1,5,9]) + randn(1,3)*sStd, SPEED_MIN), SPEED_MAX);
                % 夹紧
                pos = min(max(pos, VarMin), VarMax);
                particle(j).Position = pos;
                particle(j).Velocity = zeros(1, nVar);
                [Jr, durR, rewardR, cloud_dursR] = obj(particle(j).Position, it);
                particle(j).Cost = Jr; 
                particle(j).Duration = durR; 
                particle(j).Reward = rewardR;
                particle(j).CloudDurations = cloud_dursR;

                particle(j).Best.Position = particle(j).Position;
                particle(j).Best.Cost = Jr;
                particle(j).Best.Duration = durR;
                particle(j).Best.Reward = rewardR;
                particle(j).Best.CloudDurations = cloud_dursR;
            end
            stallCounter = 0;
        end

        w = w * wDamp;
    end

    % 返回结果
    best_position      = GlobalBest.Position;
    best_duration      = GlobalBest.Duration;
    best_reward        = GlobalBest.Reward;
    cloud_durations    = GlobalBest.CloudDurations;
    convergence_curve  = BestDuration(1:actual_iterations);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 子函数：第四问目标函数（最小化：-遮挡时长 - 退火奖赏）
function [J, duration, reward, cloud_durations] = objective_q4(x, sim_opts, ...
    v_cloud, r_cloud, r_target, h_target, pos_target, pos_M1, vv_M1, ...
    pos_FY1, pos_FY2, pos_FY3, ...
    reward_frac, reward_init, dist_scale, it, MaxIt)

    % 解包
    s1=x(1); th1=x(2); t1=x(3); dt1=x(4);
    s2=x(5); th2=x(6); t2=x(7); dt2=x(8);
    s3=x(9); th3=x(10); t3=x(11); dt3=x(12);
    vv1 = [s1*cos(th1), s1*sin(th1), 0];
    vv2 = [s2*cos(th2), s2*sin(th2), 0];
    vv3 = [s3*cos(th3), s3*sin(th3), 0];

    % 三枚云团的爆炸中心
    pos_throw_1 = pos_FY1 + t1 * vv1; pos_bao_1 = pos_throw_1 + dt1 * vv1; pos_bao_1(3) = pos_bao_1(3) - 0.5 * sim_opts.g * (dt1^2);
    pos_throw_2 = pos_FY2 + t2 * vv2; pos_bao_2 = pos_throw_2 + dt2 * vv2; pos_bao_2(3) = pos_bao_2(3) - 0.5 * sim_opts.g * (dt2^2);
    pos_throw_3 = pos_FY3 + t3 * vv3; pos_bao_3 = pos_throw_3 + dt3 * vv3; pos_bao_3(3) = pos_bao_3(3) - 0.5 * sim_opts.g * (dt3^2);

    % 多云团遮挡
    spheres = struct('startTime', {}, 'center0', {}, 'vel', {}, 'radius', {});
    spheres(1).startTime = t1+dt1; spheres(1).center0 = pos_bao_1; spheres(1).vel = [0,0,-v_cloud]; spheres(1).radius = r_cloud;
    spheres(2).startTime = t2+dt2; spheres(2).center0 = pos_bao_2; spheres(2).vel = [0,0,-v_cloud]; spheres(2).radius = r_cloud;
    spheres(3).startTime = t3+dt3; spheres(3).center0 = pos_bao_3; spheres(3).vel = [0,0,-v_cloud]; spheres(3).radius = r_cloud;

    [duration, cloud_intervals] = compute_occlusion_multi(0, sim_opts.tSimEnd, ...
        pos_M1, vv_M1, pos_target, r_target, h_target, spheres);

    % 每个云团有效遮挡时长
    cloud_durations = zeros(1, 3);
    for i = 1:3
        if ~isempty(cloud_intervals{i})
            if iscell(cloud_intervals{i})
                intervals = cell2mat(cloud_intervals{i});
            else
                intervals = cloud_intervals{i};
            end
            if ~isempty(intervals) && size(intervals, 2) >= 2
                cloud_durations(i) = sum(intervals(:,2) - intervals(:,1));
            end
        end
    end

    % 奖励：云团爆炸点到导弹轨迹的距离越近越好（Weibull型生存函数）
    d1 = point_to_line_distance(pos_bao_1, pos_M1, vv_M1);
    d2 = point_to_line_distance(pos_bao_2, pos_M1, vv_M1);
    d3 = point_to_line_distance(pos_bao_3, pos_M1, vv_M1);
    max_possible_dist = 2000;
    normalized_d = [d1 d2 d3] / max_possible_dist;
    lambda = 0.5; k_shape = 2.5;
    reward_each = exp(-((normalized_d/lambda).^k_shape));
    reward = mean(reward_each);

    % 退火权重
    reward_weight = max(0, reward_init*(1 - it/(reward_frac*MaxIt)));

    % 目标
    J = -duration - reward_weight*reward;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 子函数：多云团全局遮挡（与 Question4.m 对齐）
function [duration, cloud_intervals] = compute_occlusion_multi(tStart, tEnd, pos_M1, vv_M1, pos_target, r_target, h_target, spheres)
    duration = 0; cloud_intervals = cell(length(spheres), 1);
    all_intervals = {};
    for i = 1:length(spheres)
        sphere = spheres(i);
        init = struct( ...
            'observerPos', pos_M1, ...
            'observerVel', vv_M1, ...
            'cylinderCenter', pos_target, ...
            'cylinderVel', [0,0,0], ...
            'cylinderRadius', r_target, ...
            'cylinderHeight', h_target, ...
            'cylinderDir', [0,0,1], ...
            'sphereCenter', sphere.center0, ...
            'sphereVel', sphere.vel, ...
            'sphereRadius', sphere.radius ...
        );
        [~, intervals] = computeOcclusionSimple(tStart, tEnd, init);
        cloud_intervals{i} = intervals;
        if ~isempty(intervals) && size(intervals, 2) >= 2
            all_intervals = [all_intervals; {intervals}]; %#ok<AGROW>
        end
    end

    if isempty(all_intervals)
        duration = 0; return;
    end

    try
        valid_intervals = [];
        for j = 1:length(all_intervals)
            if ~isempty(all_intervals{j}) && size(all_intervals{j}, 2) >= 2
                valid_intervals = [valid_intervals; all_intervals{j}]; %#ok<AGROW>
            end
        end
        if ~isempty(valid_intervals)
            merged_intervals = merge_intervals(valid_intervals);
            duration = sum(merged_intervals(:,2) - merged_intervals(:,1));
        end
    catch e
        fprintf('警告: 区间合并失败，使用简单累加: %s\n', e.message);
        duration = 0;
        for j = 1:length(cloud_intervals)
            interval_j = cloud_intervals{j};
            if ~isempty(interval_j) && size(interval_j, 2) >= 2
                duration = duration + sum(interval_j(:,2) - interval_j(:,1));
            end
        end
    end
end

function merged = merge_intervals(intervals)
    if isempty(intervals), merged = []; return; end
    if size(intervals, 1) <= 1 || size(intervals, 2) < 2
        merged = intervals; return;
    end
    [~, idx] = sort(intervals(:,1)); intervals = intervals(idx,:);
    merged = intervals(1,:);
    for i = 2:size(intervals, 1)
        current = intervals(i,:); last = merged(end,:);
        if current(1) <= last(2)
            merged(end,2) = max(last(2), current(2));
        else
            merged = [merged; current]; %#ok<AGROW>
        end
    end
end

function tc = refine(ta, tb, occA, fOcc, tol)
    while (tb - ta) > tol
        tm = 0.5*(ta+tb);
        if fOcc(tm) == occA, ta = tm; else, tb = tm; end
    end
    tc = tb;
end

function d = point_to_line_distance(p, p0, v)
    % p 到直线 L: p0 + t*v 的最短距离
    d = norm(cross(v, (p - p0))) / norm(v);
end

function theta = randVonMises(mu, kappa, n)
    if kappa < 1e-8
        theta = (rand(n,1)*2*pi - pi) + mu; theta = mod(theta + pi, 2*pi) - pi; return;
    end
    a = 1 + sqrt(1 + 4*kappa^2); b = (a - sqrt(2*a)) / (2*kappa); r = (1 + b^2) / (2*b);
    theta = zeros(n,1);
    for ii = 1:n
        while true
            u1 = rand; u2 = rand; u3 = rand; z = cos(pi*u1); f = (1 + r*z) / (r + z); c = kappa * (r - f);
            if (u2 < c*(2 - c)) || (log(u2) <= (c - 1)), break; end
        end
        signv = 1; if u3 - 0.5 < 0, signv = -1; end
        theta(ii) = mu + signv * acos(f);
    end
    theta = mod(theta + pi, 2*pi) - pi;
end 