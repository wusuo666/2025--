% 粒子群优化（PSO）：第四问 - 三架无人机（FY1/2/3）各投一弹，最大化总遮挡
% 目标函数：在全局时间轴上判断任一时刻是否被任一云团遮挡，计算总遮挡时长；
% 奖励：使用云团爆炸位置到导弹轨迹的距离（越近奖励越大）

clear; clc;

%% 约束与场景常量
% 速度大小与方位角
SPEED_MIN = 70; SPEED_MAX = 140;     % s ∈ [70,140]
THETA_MIN = -pi; THETA_MAX = pi;     % theta ∈ [-pi, pi]
VM_kappa = 10;                        % 初始方位角冯米塞斯浓度

% 时间约束
T_MAX = 67;
T_SUM_MAX = 67 * (1 - 1e-6);         % 总时间上限（防数值卡边）

% 目标函数仿真选项
sim_opts = struct('tSimEnd', 20, 'g', 9.8);

% 目标/导弹/云团参数（与Q1一致，按需调整）
v_cloud   = 3;                       % 云团垂直坠落速度
r_cloud   = 10;                      % 云团半径
r_target  = 7;                       % 目标半径
h_target  = 10;                      % 目标高度
pos_target = [0, 200, 5];
pos_fake = [0,0,0];
pos_M1     = [20000, 0, 2000];
v_M       = 300;                     % 导弹速度
vv_M1     = v_M * (pos_fake - pos_M1) / norm(pos_fake - pos_M1);

% 三架无人机初始位置（请按实际题面调整）
pos_FY1    = [17800, 0, 1800];
pos_FY2    = [12000, 1400, 1400];
pos_FY3    = [6000,  -3000, 700];

%% 奖赏机制参数（退火）
REWARD_MAX_ITER_FRAC = 0.5;         % 前50%迭代使用明显奖励
REWARD_WEIGHT_INIT   = 40;          % 初始奖赏权重
DIST_SCALE           = 300;          % 距离尺度（m），越小越强调接近导弹轨迹

%% 粒子群参数与搜索边界
% 变量（12维）：[s1 th1 t1 dt1  s2 th2 t2 dt2  s3 th3 t3 dt3]
nVar = 12;
VarMin = [SPEED_MIN, THETA_MIN, 0, 0,  SPEED_MIN, THETA_MIN, 0, 0,  SPEED_MIN, THETA_MIN, 0, 0];
VarMax = [SPEED_MAX, THETA_MAX, T_MAX, T_MAX,  SPEED_MAX, THETA_MAX, T_MAX, T_MAX,  SPEED_MAX, THETA_MAX, T_MAX, T_MAX];

nPop = 150;              % 群体规模
MaxIt = 100;             % 最大迭代次数
w = 1.2;              % 惯性权重
wDamp = 0.99;            % 惯性权重衰减
c1 = 2.0;                % 个体学习因子
c2 = 1.6;                % 群体学习因子

% 速度上限（按搜索区间比例）
VelMax = 0.25 * (VarMax - VarMin);
VelMin = -VelMax;

%% 目标函数封装（最小化：-遮挡时长 - 奖励项）
obj = @(x, it) objective_q4(x, sim_opts, ...
    v_cloud, r_cloud, r_target, h_target, pos_target, pos_M1, vv_M1, ...
    pos_FY1, pos_FY2, pos_FY3, ...
    REWARD_MAX_ITER_FRAC, REWARD_WEIGHT_INIT, DIST_SCALE, it, MaxIt);

%% 粒子结构体
empty_particle.Position = [];
empty_particle.Velocity = [];
empty_particle.Cost     = [];
empty_particle.Duration = [];
empty_particle.Reward   = 0;
empty_particle.CloudDurations = [];
empty_particle.Best = empty_particle;

% 初始化群体
particle  = repmat(empty_particle, nPop, 1);
GlobalBest.Cost = inf; GlobalBest.Duration = -inf; GlobalBest.Reward = -inf; GlobalBest.Position = [];
GlobalBest.CloudDurations = [];

for i = 1:nPop
    particle(i).Position = VarMin + rand(1, nVar) .* (VarMax - VarMin);




    % FY1的方位角：使用冯米塞斯分布指向真实目标
    if rand < 0.5
        mu_sample = 0;
    else
        mu_sample = pi;
    end
    particle(i).Position(2) = randVonMises(mu_sample, VM_kappa, 1);             % theta ∈ [-pi,pi]

    % FY2的方位角：指向导弹和目标连线中点，k=5
    midpoint = (pos_M1 + pos_target) / 2;  % 导弹和目标连线的中点
    midpoint_direction = midpoint - pos_FY2;
    midpoint_theta = atan2(midpoint_direction(2), midpoint_direction(1));
    particle(i).Position(6) = randVonMises(midpoint_theta, 2.5, 1);  % k=2.5

    % FY3的方位角：单边冯米塞斯分布，k=5
    target_direction_FY3 = pos_target - pos_FY3;
    target_theta_FY3 = atan2(target_direction_FY3(2), target_direction_FY3(1));

    % 生成标准冯米塞斯分布的样本
    theta_raw = randVonMises(0, 5, 1);  % 中心在0
    % 取绝对值使其变成单边分布
    theta_abs = abs(theta_raw);
    % 添加到目标方向
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
BestCost = nan(MaxIt,1); BestDuration = nan(MaxIt,1); BestReward = nan(MaxIt,1);

% 停滞检测（多样性注入）
stallThreshold = 8; improveTol = 1e-6; stallCounter = 0; prevBestCost = inf;

%% 迭代优化
for it = 1:MaxIt

    for i = 1:nPop
        r1 = rand(1, nVar); 
        r2 = rand(1, nVar);
        particle(i).Velocity = w*particle(i).Velocity ...
            + c1*r1.*(particle(i).Best.Position - particle(i).Position) ...
            + c2*r2.*(GlobalBest.Position - particle(i).Position);
        particle(i).Velocity = min(max(particle(i).Velocity, VelMin), VelMax);
        particle(i).Position = particle(i).Position + particle(i).Velocity;
        particle(i).Position = min(max(particle(i).Position, VarMin), VarMax);
        % 角度wrap
        particle(i).Position([2,6,10]) = arrayfun(@wrapToPi, particle(i).Position([2,6,10]));

        % 重新评估
        [J, dur, rew, cloud_durs] = obj(particle(i).Position, it);
        particle(i).Cost = J; 
        particle(i).Duration = dur; 
        particle(i).Reward = rew;
        particle(i).CloudDurations = cloud_durs;

        % 个体最优
        if J < particle(i).Best.Cost
            particle(i).Best = particle(i);
        end

        % 全局最优（前期考虑奖励并列）
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

    BestCost(it) = GlobalBest.Cost; 
    BestDuration(it) = GlobalBest.Duration; 
    BestReward(it) = GlobalBest.Reward;

    % 停滞检测与注入
    if it == 1 || (BestCost(it) < prevBestCost - improveTol)
        prevBestCost = BestCost(it); stallCounter = 0;
    else
        stallCounter = stallCounter + 1;
    end
    if stallCounter >= stallThreshold
        numReseed = max(1, round(0.50 * nPop));
        costs = [particle.Cost]; [~, idxSorted] = sort(costs, 'descend'); reseedIdx = idxSorted(1:numReseed);
        for k = 1:numReseed
            j = reseedIdx(k); pos = particle(j).Position;
            % 随机化（均匀重新采样）
            pos(1:12) = VarMin(1:12) + rand(1,12).*(VarMax(1:12)-VarMin(1:12));
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
    % 惯性权重衰减
    w = w * wDamp;
    fprintf('Iter %3d | BestDuration = %.6f s | Cost = %.6f | Reward = %.6f\n', it, GlobalBest.Duration, GlobalBest.Cost, GlobalBest.Reward);
end

%% 输出最优解
xbest = GlobalBest.Position;
[s1,th1,t1,dt1, s2,th2,t2,dt2, s3,th3,t3,dt3] = deal(xbest(1),xbest(2),xbest(3),xbest(4), xbest(5),xbest(6),xbest(7),xbest(8), xbest(9),xbest(10),xbest(11),xbest(12));
[vx1,vy1] = deal(s1*cos(th1), s1*sin(th1));
[vx2,vy2] = deal(s2*cos(th2), s2*sin(th2));
[vx3,vy3] = deal(s3*cos(th3), s3*sin(th3));

fprintf('\n最优解：\n');
fprintf('  FY1: s=%.3f, th=%.3f, t=%.3f, dt=%.3f\n', s1, th1, t1, dt1);
fprintf('  FY2: s=%.3f, th=%.3f, t=%.3f, dt=%.3f\n', s2, th2, t2, dt2);
fprintf('  FY3: s=%.3f, th=%.3f, t=%.3f, dt=%.3f\n', s3, th3, t3, dt3);

totalDuration = GlobalBest.Duration;
fprintf('\n  最优遮挡总时长 = %.6f s\n', totalDuration);

% 输出各个云团的遮挡贡献
if ~isempty(GlobalBest.CloudDurations)
    fprintf('\n各云团的遮挡贡献：\n');
    fprintf('  云团1遮挡时长: %.2f秒\n', GlobalBest.CloudDurations(1));
    fprintf('  云团2遮挡时长: %.2f秒\n', GlobalBest.CloudDurations(2));
    fprintf('  云团3遮挡时长: %.2f秒\n', GlobalBest.CloudDurations(3));
end

% 根据最优解计算云团位置
xbest = GlobalBest.Position;
s1_best=xbest(1); th1_best=xbest(2); t1_best=xbest(3); dt1_best=xbest(4);
s2_best=xbest(5); th2_best=xbest(6); t2_best=xbest(7); dt2_best=xbest(8);
s3_best=xbest(9); th3_best=xbest(10); t3_best=xbest(11); dt3_best=xbest(12);

vv1_best = [s1_best*cos(th1_best), s1_best*sin(th1_best), 0];
vv2_best = [s2_best*cos(th2_best), s2_best*sin(th2_best), 0];
vv3_best = [s3_best*cos(th3_best), s3_best*sin(th3_best), 0];

% 计算三枚云团的爆炸中心（各自引爆瞬间）
pos_throw_1 = pos_FY1 + t1_best * vv1_best; 
pos_bao_1 = pos_throw_1 + dt1_best * vv1_best; 
pos_bao_1(3) = pos_bao_1(3) - 0.5 * sim_opts.g * (dt1_best^2);

pos_throw_2 = pos_FY2 + t2_best * vv2_best; 
pos_bao_2 = pos_throw_2 + dt2_best * vv2_best; 
pos_bao_2(3) = pos_bao_2(3) - 0.5 * sim_opts.g * (dt2_best^2);

pos_throw_3 = pos_FY3 + t3_best * vv3_best; 
pos_bao_3 = pos_throw_3 + dt3_best * vv3_best; 
pos_bao_3(3) = pos_bao_3(3) - 0.5 * sim_opts.g * (dt3_best^2);

%% 子函数：目标函数
function [J, duration, reward, cloud_durations] = objective_q4(x, sim_opts, ...
    v_cloud, r_cloud, r_target, h_target, pos_target, pos_M1, vv_M1, ...
    pos_FY1, pos_FY2, pos_FY3, ...
    reward_frac, reward_init, dist_scale, it, MaxIt)

    % 解包变量
    s1=x(1); th1=x(2); t1=x(3); dt1=x(4);
    s2=x(5); th2=x(6); t2=x(7); dt2=x(8);
    s3=x(9); th3=x(10); t3=x(11); dt3=x(12);
    vv1 = [s1*cos(th1), s1*sin(th1), 0];
    vv2 = [s2*cos(th2), s2*sin(th2), 0];
    vv3 = [s3*cos(th3), s3*sin(th3), 0];

    % 计算三枚云团的爆炸中心（各自引爆瞬间）
    pos_throw_1 = pos_FY1 + t1 * vv1; pos_bao_1 = pos_throw_1 + dt1 * vv1; pos_bao_1(3) = pos_bao_1(3) - 0.5 * sim_opts.g * (dt1^2);
    pos_throw_2 = pos_FY2 + t2 * vv2; pos_bao_2 = pos_throw_2 + dt2 * vv2; pos_bao_2(3) = pos_bao_2(3) - 0.5 * sim_opts.g * (dt2^2);
    pos_throw_3 = pos_FY3 + t3 * vv3; pos_bao_3 = pos_throw_3 + dt3 * vv3; pos_bao_3(3) = pos_bao_3(3) - 0.5 * sim_opts.g * (dt3^2);

    % 计算全局遮挡区间与总时长 - 使用q1_occlusion_time_flex
    opts = struct('tSimEnd', sim_opts.tSimEnd, 'g', sim_opts.g, ...
                 'target_pos', pos_target, 'r_target', r_target, 'h_target', h_target, ...
                 'v_cloud', v_cloud, 'r_cloud', r_cloud);
    
    % 计算每个云团的遮挡区间（全局时间轴）
    cloud_durations = zeros(1, 3);
    cloud_intervals = cell(1, 3);
    
    % 云团1
    [cloud_durations(1), intervals1] = q1_occlusion_time_flex(vv1(1), vv1(2), t1, dt1, pos_M1, pos_FY1, opts);
    if ~isempty(intervals1)
        % 将相对时间转为全局时间（相对爆炸时刻 -> 全局时间轴）
        cloud_intervals{1} = intervals1 + (t1 + dt1);
    else
        cloud_intervals{1} = [];
    end
    
    % 云团2
    [cloud_durations(2), intervals2] = q1_occlusion_time_flex(vv2(1), vv2(2), t2, dt2, pos_M1, pos_FY2, opts);
    if ~isempty(intervals2)
        cloud_intervals{2} = intervals2 + (t2 + dt2);
    else
        cloud_intervals{2} = [];
    end
    
    % 云团3
    [cloud_durations(3), intervals3] = q1_occlusion_time_flex(vv3(1), vv3(2), t3, dt3, pos_M1, pos_FY3, opts);
    if ~isempty(intervals3)
        cloud_intervals{3} = intervals3 + (t3 + dt3);
    else
        cloud_intervals{3} = [];
    end
    
    % 合并所有区间计算总遮挡时长
    valid_intervals = [];
    for i = 1:3
        if ~isempty(cloud_intervals{i}) && size(cloud_intervals{i}, 2) >= 2
            valid_intervals = [valid_intervals; cloud_intervals{i}];
        end
    end
    
    if isempty(valid_intervals)
        duration = 0;
    else
        % 合并重叠区间
        merged_intervals = merge_time_intervals(valid_intervals);
        duration = sum(merged_intervals(:,2) - merged_intervals(:,1));
    end

    
    % 奖励：云团爆炸点到"爆炸时刻下的导弹与真目标连线"的距离（使用Weibull CDF）
    % 计算爆炸时刻下各导弹位置
    time_bao_1 = t1 + dt1;
    time_bao_2 = t2 + dt2;
    time_bao_3 = t3 + dt3;
    
    % 计算各爆炸时刻下导弹位置
    missile_pos_1 = pos_M1 + time_bao_1 * vv_M1;
    missile_pos_2 = pos_M1 + time_bao_2 * vv_M1;
    missile_pos_3 = pos_M1 + time_bao_3 * vv_M1;
    
    % 创建导弹-目标连线向量
    missile_target_vec_1 = pos_target - missile_pos_1;
    missile_target_vec_2 = pos_target - missile_pos_2;
    missile_target_vec_3 = pos_target - missile_pos_3;
    
    % 计算爆炸点到导弹-目标连线的距离
    d1 = point_to_line_distance(pos_bao_1, missile_pos_1, missile_target_vec_1);
    d2 = point_to_line_distance(pos_bao_2, missile_pos_2, missile_target_vec_2);
    d3 = point_to_line_distance(pos_bao_3, missile_pos_3, missile_target_vec_3);

    % 归一化距离（假设最大可能距离为1000m）
    max_possible_dist = 100;  % 根据您的场景调整
    normalized_d1 = d1 / max_possible_dist;
    normalized_d2 = d2 / max_possible_dist;
    normalized_d3 = d3 / max_possible_dist;

    % 使用Weibull分布计算奖赏
    % Weibull CDF: F(x) = 1 - exp(-(x/λ)^k)
    % 为保持距离小时奖赏高，使用生存函数 S(x) = 1 - F(x) = exp(-(x/λ)^k)
    lambda = 0.5;  % 尺度参数
    k_shape = 2.5;  % 形状参数
    reward_each = exp(-((([normalized_d1, normalized_d2, normalized_d3])/lambda).^k_shape));
    reward = mean(reward_each);
    
    % 迭代退火权重（阶跃式变化）
    if it <= reward_frac*MaxIt
        reward_weight = reward_init;
    else
        reward_weight = 0;
    end

    % 目标函数（最小化）
    J = -duration - reward_weight*reward;
end


%% 子函数：合并时间区间（处理重叠）
function merged = merge_time_intervals(intervals)
    % 合并可能重叠的时间区间
    % 输入: intervals - n×2的矩阵，每行为[start_time, end_time]
    % 输出: merged - 合并后的不重叠区间
    
    if isempty(intervals)
        merged = [];
        return;
    end
    
    % 检查输入维度
    if size(intervals, 2) < 2
        merged = intervals;
        return;
    end
    
    % 按开始时间排序
    [~, idx] = sort(intervals(:,1));
    sorted_intervals = intervals(idx,:);
    
    % 初始化合并结果为第一个区间
    merged = sorted_intervals(1,:);
    
    % 遍历并合并重叠区间
    for i = 2:size(sorted_intervals, 1)
        current = sorted_intervals(i,:);
        last = merged(end,:);
        
        % 检查是否重叠 (current.start <= last.end)
        if current(1) <= last(2)
            % 重叠，更新结束时间为较大值
            merged(end,2) = max(last(2), current(2));
        else
            % 不重叠，添加新区间
            merged = [merged; current];
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

%% 子函数：点到直线距离
function d = point_to_line_distance(p, p0, v)
    % p 到 直线 L: p0 + t*v 的最短距离
    d = norm(cross(v, (p - p0))) / norm(v);
end

%% 子函数：von Mises 采样（仅用于微扰角度，mu=0时即对称扰动）
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
