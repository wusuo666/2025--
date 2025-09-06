% 粒子群优化（PSO）：最大化第三问遮挡总时长（3次投弹）
% 目标函数：计算3次投弹后的总遮挡时长，并设计奖赏机制

clear; clc;

%% 约束与场景常量
% 速度大小与方位角
SPEED_MIN = 70; SPEED_MAX = 140;     % s ∈ [70,140]
THETA_MIN = -pi; THETA_MAX = pi;     % theta ∈ [-pi, pi]

VM_kappa = 10;     % 浓度参数（恒定）

% 时间约束
T_MAX = 20;
T_SUM_MAX = 20 * (1 - 1e-6);        % 略微留余量，保证严格小于
MIN_INTERVAL = 1;                    % 两次投弹之间的最小时间间隔

% 目标函数仿真选项
q1_opts = struct('tSimEnd', 20, 'g', 9.8);

% 奖赏机制参数
REWARD_MAX_ITER_FRAC = 0.5;         % 前50%的迭代使用奖赏
REWARD_WEIGHT_INIT = 1;           % 初始奖赏权重

%% 粒子群参数与搜索边界
% 变量：[s theta t1 dt1 t2 dt2 t3 dt3]
% t1,t2,t3为三次投弹时刻，dt1,dt2,dt3为从投弹到引爆的时间
nVar = 8;
VarMin = [SPEED_MIN, THETA_MIN, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
VarMax = [SPEED_MAX, THETA_MAX, T_MAX, T_MAX, T_MAX, T_MAX, T_MAX, T_MAX];

nPop = 120;              % 群体规模
MaxIt = 60;             % 最大迭代次数
w = 1.2;                % 惯性权重
wDamp = 0.99;            % 惯性权重衰减
c1 = 1.6;                % 个体学习因子
c2 = 2.0;                % 群体学习因子

% 速度上限（按搜索区间的一定比例设定）
VelMax = 0.25 * (VarMax - VarMin);
VelMin = -VelMax;

%% 目标函数（最小化：-遮挡时长；带奖赏机制）
obj = @(x, it, MaxIt) objective_function(x, q1_opts, it, MaxIt, ...
    REWARD_MAX_ITER_FRAC, REWARD_WEIGHT_INIT);

%% 粒子结构体
empty_particle.Position = [];
empty_particle.Velocity = [];
empty_particle.Cost = [];
empty_particle.Duration = [];
empty_particle.Reward = 0;          % 新增：奖赏值
empty_particle.Best.Position = [];
empty_particle.Best.Cost = [];
empty_particle.Best.Duration = [];
empty_particle.Best.Reward = 0;     % 新增：奖赏值

% 初始化群体
particle = repmat(empty_particle, nPop, 1);
GlobalBest.Cost = inf;
GlobalBest.Duration = -inf;
GlobalBest.Reward = -inf;
GlobalBest.Position = [];

% 导弹初始速度方向（用于奖赏计算）
pos_M1     = [20000, 0, 2000];      % 导弹起始位置
pos_fake   = [0, 0, 0];             % 假目标位置
v_M = 300;                          % 导弹速度
vv_M1 = v_M * (pos_fake - pos_M1) / norm(pos_fake - pos_M1);  % 导弹速度向量

for i = 1:nPop
    % 初始化位置
    particle(i).Position = VarMin + rand(1, nVar) .* (VarMax - VarMin);
    
    % 方位角：使用冯米塞斯分布指向真实目标
    pos_target = [0, 200, 0];  % 真实目标位置
    pos_FY1 = [17800, 0, 1800]; % FY1初始位置
    target_direction = pos_target - pos_FY1;
    target_theta = atan2(target_direction(2), target_direction(1)); % 计算指向目标的方位角
    particle(i).Position(2) = randVonMises(target_theta, VM_kappa, 1); % 围绕目标方向的冯米塞斯分布
    
    % 修复投弹时间约束
    particle(i).Position = fix_time_constraints(particle(i).Position, MIN_INTERVAL, T_SUM_MAX);
    
    particle(i).Velocity = zeros(1, nVar);

    % 计算初始目标函数值
    [J, dur, reward] = obj(particle(i).Position, 1, MaxIt);
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
BestCost = nan(MaxIt,1);
BestDuration = nan(MaxIt,1);
BestReward = nan(MaxIt,1);

% 停滞检测与多样性注入参数
stallThreshold = 12;          % 连续无改进的阈值（迭代数）
improveTol = 1e-6;            % 认为有改进的最小变化量
stallCounter = 0;             % 当前停滞计数
prevBestCost = inf;           % 上一次有改进时的BestCost

%% 迭代优化
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

        % 位置越界处理（饱和 + 角度环绕）
        particle(i).Position = min(max(particle(i).Position, VarMin), VarMax);
        particle(i).Position(2) = wrapToPi(particle(i).Position(2)); % theta wrap
        
        % 修复投弹时间约束
        particle(i).Position = fix_time_constraints(particle(i).Position, MIN_INTERVAL, T_SUM_MAX);

        % 重新计算适应度
        [J, dur, reward] = obj(particle(i).Position, it, MaxIt);
        particle(i).Cost = J;
        particle(i).Duration = dur;
        particle(i).Reward = reward;

        % 更新个体最优（基于总成本）
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

    % 记录迭代结果
    BestCost(it) = GlobalBest.Cost;
    BestDuration(it) = GlobalBest.Duration;
    BestReward(it) = GlobalBest.Reward;

    % 停滞检测：统计BestCost是否改进
    if it == 1 || (BestCost(it) < prevBestCost - improveTol)
        prevBestCost = BestCost(it);
        stallCounter = 0;
    else
        stallCounter = stallCounter + 1;
    end

    % 多样性注入：若长期无改进，则扰动/重置最差10%粒子的时间维
    if stallCounter >= stallThreshold
        numReseed = max(1, round(0.50 * nPop));
        % 根据当前Cost从差到好排序（Cost越大越差）
        costs = [particle.Cost];
        [~, idxSorted] = sort(costs, 'descend');
        reseedIdx = idxSorted(1:numReseed);
        
        for k = 1:numReseed
            j = reseedIdx(k);
            pos = particle(j).Position;
            
            % 时间维度随机化（均匀重新采样）
            pos(3:8) = VarMin(3:8) + rand(1,6).*(VarMax(3:8)-VarMin(3:8));
            
            % 约束修复（确保顺序与最小间隔）
            pos = fix_time_constraints(pos, MIN_INTERVAL, T_SUM_MAX);
            
            % 对方向(theta)与速度大小(s)加入高斯扰动
            % theta 扰动（约20度的标准差）
            thetaStd = deg2rad(20);
            pos(2) = wrapToPi(pos(2) + randn * thetaStd);
            % s 扰动（按区间范围的8%作为标准差）
            pos(1) = pos(1) + randn * 10;
            pos(1) = min(max(pos(1), SPEED_MIN), SPEED_MAX);
            
            % 覆盖粒子位置并清零速度（避免过大动量）
            particle(j).Position = pos;
            particle(j).Velocity = zeros(1, nVar);
            
            % 重新评估并重置个体最优为当前，鼓励探索
            [Jr, durR, rewardR] = obj(particle(j).Position, it, MaxIt);
            particle(j).Cost = Jr;
            particle(j).Duration = durR;
            particle(j).Reward = rewardR;
            
            particle(j).Best.Position = particle(j).Position;
            particle(j).Best.Cost = Jr;
            particle(j).Best.Duration = durR;
            particle(j).Best.Reward = rewardR;
        end
        
        % 重置停滞计数
        stallCounter = 0;
    end

    % 衰减惯性权重
    w = w * wDamp;

    % 迭代日志
    fprintf('Iter %3d | BestDuration = %.6f s | Cost = %.6f | Reward = %.6f\n', ...
        it, GlobalBest.Duration, GlobalBest.Cost, GlobalBest.Reward);
end

%% 输出最优解
s_best = GlobalBest.Position(1);
th_best = GlobalBest.Position(2);
t1_best = GlobalBest.Position(3);
dt1_best = GlobalBest.Position(4);
t2_best = GlobalBest.Position(5);
dt2_best = GlobalBest.Position(6);
t3_best = GlobalBest.Position(7);
dt3_best = GlobalBest.Position(8);

vx_best = s_best * cos(th_best);
vy_best = s_best * sin(th_best);

% 计算三次投弹的总遮挡时长
[totalDuration, allIntervals] = calculate_total_occlusion(...
    vx_best, vy_best, [t1_best, t2_best, t3_best], [dt1_best, dt2_best, dt3_best], q1_opts);

fprintf('\n最优解：\n');
fprintf('  s (速度大小) = %.6f m/s\n', s_best);
fprintf('  theta (方位角) = %.6f rad\n', th_best);
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

fprintf('\n  最优遮挡总时长 = %.6f s\n', totalDuration);
if ~isempty(allIntervals)
    disp('  遮挡区间 [t_begin, t_end] (s):');
    disp(allIntervals);
end

%% 可视化收敛曲线
figure('Name','PSO 收敛曲线');
subplot(3,1,1); plot(BestDuration,'LineWidth',1.5); grid on;
ylabel('遮挡时长 (s)'); xlabel('迭代次数'); title('最佳遮挡时长');

subplot(3,1,2); plot(BestReward,'LineWidth',1.5); grid on;
ylabel('奖赏值'); xlabel('迭代次数'); title('奖赏值变化');

subplot(3,1,3); plot(BestCost,'LineWidth',1.5); grid on;
ylabel('目标函数值'); xlabel('迭代次数'); title('目标函数收敛曲线');

%% 辅助函数：修复时间约束
function pos = fix_time_constraints(pos, min_interval, t_sum_max)
    % 1. 确保时间顺序和最小间隔
    t1 = pos(3); dt1 = pos(4);
    t2 = pos(5); dt2 = pos(6);
    t3 = pos(7); dt3 = pos(8);
    
    % 按照大小排序三个投弹时间
    times = [t1, t2, t3];
    [sorted_times, indices] = sort(times);
    
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

%% 辅助函数：计算总遮挡时长
function [totalDuration, mergedIntervals] = calculate_total_occlusion(vx, vy, t_throws, t_explodes, opts)
    % 分别计算每次投弹产生的遮挡区间
    numBombs = length(t_throws);
    allIntervals = [];
    
    % 计算每次投弹的遮挡区间
    for i = 1:numBombs
        [dur, intervals] = q1_occlusion_time(vx, vy, t_throws(i), t_explodes(i), opts);
        if ~isempty(intervals)
            offset = t_throws(i) + t_explodes(i); % 将相对引爆时刻的区间偏移到全局时间轴
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

%% 目标函数：从 [s,theta,t1,dt1,t2,dt2,t3,dt3] 计算总遮挡（带奖赏机制）
function [J, duration, reward] = objective_function(x, q1_opts, it, MaxIt, ...
                                                   REWARD_MAX_ITER_FRAC, REWARD_WEIGHT_INIT)
    % 提取参数
    s = x(1); th = x(2);
    t1 = x(3); dt1 = x(4);
    t2 = x(5); dt2 = x(6);
    t3 = x(7); dt3 = x(8);
    
    % 计算速度分量
    vx = s * cos(th); vy = s * sin(th);
    
    % 计算总遮挡时长
    try
        [duration, ~] = calculate_total_occlusion(vx, vy, [t1, t2, t3], [dt1, dt2, dt3], q1_opts);
    catch
        duration = 0;
    end
    
    % 计算方向奖赏
    reward = 0;
    
    % 计算FY1方向与导弹方向的夹角
    pos_M1 = [20000, 0, 2000];  % 导弹起始位置
    pos_target = [0, 200, 0];   % 真实目标位置
    pos_FY1 = [17800, 0, 1800]; % FY1初始位置
    
    % 计算导弹指向真实目标的方向向量
    v_M = 300;
    vv_M1 = v_M * (pos_target - pos_M1) / norm(pos_target - pos_M1);  % 导弹指向真实目标的速度向量
    vv_FY1 = [vx, vy, 0];  % FY1速度向量
    
    % 计算夹角
    cos_angle = dot(vv_FY1, vv_M1) / (norm(vv_FY1) * norm(vv_M1));
    angle = acos(max(min(cos_angle, 1), -1));  % 防止数值误差
    
    % 使用标准Weibull累积分布函数(CDF)计算奖赏
    % 将角度映射到[0,1]区间
    normalized_angle = angle / pi;
    
    % Weibull CDF: F(x) = 1 - exp(-(x/λ)^k)
    % 为保持方向一致时奖赏高，使用生存函数 S(x) = 1 - F(x) = exp(-(x/λ)^k)
    lambda = 1;  % 尺度参数
    k_shape = 3.5;  % 形状参数
    reward = exp(-((normalized_angle/lambda)^k_shape));
    
    % 随迭代逐渐减小奖赏权重（确保不为负值）
    reward_weight = max(0, REWARD_WEIGHT_INIT * (1 - it / (REWARD_MAX_ITER_FRAC * MaxIt)));
    
    % 最终目标函数 = -遮挡时长 - reward_weight * reward
    % 当迭代次数超过REWARD_MAX_ITER_FRAC * MaxIt时，reward_weight为0，等价于仅优化遮挡时长
    J = -duration - reward_weight * reward;
end

%% von Mises 随机采样（Best & Fisher 算法）
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