clear

%% Model parameters
% parameters are randomly selected, the only 
% consideration is that A cannot have large eigenvalues
% otherwise,Yp Yf would be large and the optimization solver has numerical issues


n=4;
m=3;
p=2;

% A=2*rand(n,n)-1;

A = [0.6799   -0.0331   -0.8332    0.4924;
     0.9748    1.0060    0.3666    0.5863;
     0.7311    0.3693   -1.0711    0.1603;
    -0.7442    0.0330    0.0667    0.1961];

% B=2*rand(n,m)-1;

B = [-0.7841   -0.1798   -0.0757;
      0.5204   -0.5806   -0.6510;
      0.1974    0.2140   -0.4851;
     -0.9378    0.7881   -0.1826];

% C=2*rand(p,n)-1;

C = [0.4458    0.4911    0.7394   -0.1359
     0.0733   -0.1468   -0.6357    0.7353];

maxeig = max(abs(eig(A)))

% A = A/maxeig*1.0173;

% if maxeig>1.05
%     A = A/maxeig*(0.899+0.15*rand);
% end

max(abs(eig(A)))
while true
%% Generate noiseless Up Yp Uf Yf
Td=110;   % Number of historical data points
Te=20;    % Horizon length
Tini = n+2;   % Tini > lag(system) should hold
% x=zeros(3,T+1);
x0 = rand(n,1);
u=1*(2*rand(m*Td,1)-1);

% y=zeros(p*T,1);

O = zeros(n*Td,n);
H = zeros(n*Td,m*Td);
HC = zeros(p*Td,m*Td);
for t = 1:Td
    O((t-1)*n+1:t*n,:) = A^(t-1);
    for k = 1:t-1
        H((t-1)*n+1:t*n,(k-1)*m+1:k*m) = A^(t-k-1)*B;
        HC((t-1)*p+1:t*p,(k-1)*m+1:k*m) = C*A^(t-k-1)*B;
    end
end

OC = kron(eye(Td),C)*O;
% HC
 
% for t=1:T
%     x(:,t+1)=A*x(:,t)+B*u(t);
%     y(t)=C*x(:,t);
% end

x = O*x0+H*u;
y = OC*x0+HC*u;

% construct hankel matrix from u and y
Utotal=zeros((Tini+Te)*m,Td-(Tini+Te)+1);
Ytotal=zeros((Tini+Te)*p,Td-(Tini+Te)+1);
for i=1:size(Utotal,2)
    Utotal(:,i)=u((i-1)*m+1:(i-1)*m+(Tini+Te)*m);
    Ytotal(:,i)=y((i-1)*p+1:(i-1)*p+(Tini+Te)*p);
%     Utotal(i,:)=u(i:30+i);
%     Ytotal(i,:)=y(i:30+i);
end

% Up,Yp,Uf,Yf 
Up=Utotal(1:Tini*m,:);
Yp=Ytotal(1:Tini*p,:);
Uf=Utotal(Tini*m+1:end,:);
Yf=Ytotal(Tini*p+1:end,:);

%% Generate noisy uini yini
%  WARNING: a
%  uini -> the DISTURBANCE-FREE input sequence (what we use in calculations)
%  yini -> the NOISY output sequence (what we use in calculations)

% bounded disturbance d with \|d\|^2\le \epsilon_d

epsilon_d=0.001;
dini=2*sqrt(epsilon_d)*rand(Tini*m,1)-sqrt(epsilon_d);
d=2*sqrt(epsilon_d)*rand(Te*m,1)-sqrt(epsilon_d);

% bounded measurement noise rho with \|rho\|^2\le \epsilon_rho

epsilon_rho=0.001;
rho=2*sqrt(epsilon_rho)*rand(Tini*p,1)-sqrt(epsilon_rho);

% x=zeros(3,6);
uini=2*rand(Tini*m,1)-1;  % Noise-free (disturbance-free) input 
% y=zeros(n*p,1);


Os = O(1:Tini*n,:);
Hs = H(1:Tini*n,1:Tini*m);
OCs = OC(1:Tini*p,:);
HCs = HC(1:Tini*p,1:Tini*m);

xini=rand(n,1);
load('exp1_init.mat')
% x(:,1)=xini;

% for t=1:5
%     x(:,t+1)=A*x(:,t)+B*u(t);
%     y(t)=C*x(:,t)+w(t);
% end

x = Os*xini + Hs*(uini+dini);
yini = OCs*xini + HCs*(uini+dini) + rho;
% yini == kron(eye(Tini),C)*x+rho

%% Construct constraint specification

%  d and rho have quadratic constraints:
%  [1,dbar']*Phi_d*[1;dbar] \geq 0
%  [1,w']*Phi_w*[1;w] \geq 0
%  Both these two constraints are equivalent to quadratic constraints on    
%  rho = [dbar;w]

% Phi_d = [Phi_d1 Phi_d2; Phi_d2' Phi_d3]
% Phi_w = [Phi_w1 Phi_w2; Phi_w2' Phi_w3]

Phi_w1=Tini*p*epsilon_rho;
Phi_w2=zeros(1,Tini*p);
Phi_w3=-eye(Tini*p);

Phi_d1=(Tini+Te)*m*epsilon_d;
Phi_d2=zeros(1,(Tini+Te)*m);
Phi_d3=-eye((Tini+Te)*m);

% rho = [dbar;w] = barA1*gbar + barA2
% dbar = barA1d*gbar + barA2d
% w = barA1w*gbar + barA2w
% where gbar = [g0star; d; g0]
% Now, g0star is not a known variable! It solves Up*g0star = uini+d where d
% is not known.
% Moreover, g0 is such that Yp*g1 = yini-w where g1 = M*g0 + Mstar*g0star

Ncal=null(Up);
M = Ncal*orth(Ncal'*Yp');

% UpYp = [Up;Yp];
% M = orth(UpYp')*null(Up*orth(UpYp'));

Md = orth(Up');

barA1d=[Up*Md                 zeros(Tini*m,Te*m)     zeros(Tini*m,size(M,2));
        zeros(Te*m,rank(Up))     eye(Te*m)            zeros(Te*m,size(M,2)) ];
barA1w=[-Yp*Md                zeros(Tini*p,Te*m)          -Yp*M             ];
barA2d=[-uini       ;
        zeros(Te*m,1)];
barA2w=yini;
 
% Then, the quadratic constraints on d and rho can be transformed into
% quadratic constraints on gbar as [1,gbar']*barPhiw*[1;gbar] \geq 0 and 
% [1,gbar']*barPhid*[1;gbar] \geq 0 with the barPhiw and barPhid matrices below:

barPhid_11 = Phi_d1 + Phi_d2*barA2d + barA2d'*Phi_d2' + barA2d'*Phi_d3*barA2d;
barPhid_12 = Phi_d2*barA1d + barA2d'*Phi_d3*barA1d;
barPhid_22 = barA1d'*Phi_d3*barA1d;

barPhiw_11 = Phi_w1 + Phi_w2*barA2w + barA2w'*Phi_w2' + barA2w'*Phi_w3*barA2w;
barPhiw_12 = Phi_w2*barA1w + barA2w'*Phi_w3*barA1w;
barPhiw_22 = barA1w'*Phi_w3*barA1w;

barPhid = [barPhid_11  barPhid_12;
           barPhid_12' barPhid_22];
barPhiw = [barPhiw_11  barPhiw_12;
           barPhiw_12' barPhiw_22];
 
% A31_11=Phi_d1-Phi_d2*uini-uini'*Phi_d2'+uini'*Phi_d3*uini;
% A31_12=Phi_d2*Up-uini'*Phi_d3*Up;
% A31_21=Up'*Phi_d2'-Up'*Phi_d3*uini;
% A31_22=Up'*Phi_d3*Up;
% 
% A31=[A31_11            , A31_12                     , zeros(1,size(M,2))         ;
%      A31_21            , A31_22                     , zeros(size(Up,2),size(M,2));
%      zeros(size(M,2),1), zeros(size(M,2),size(Up,2)), zeros(size(M,2))           ];
% 
% A32_11=Phi_w1+Phi_w2*yini+yini'*Phi_w2'+yini'*Phi_w3*yini;
% A32_12=-Phi_w2*Yp-yini'*Phi_w3*Yp;
% A32_13=-Phi_w2*Yp*M-yini'*Phi_w3*Yp*M;
% A32_22=Yp'*Phi_w3*Yp;
% A32_23=Yp'*Phi_w3*Yp*M;
% A32_33=M'*Yp'*Phi_w3*Yp*M;
% 
% A32=[A32_11 , A32_12 , A32_13;
%      A32_12', A32_22 , A32_23;
%      A32_13', A32_23', A32_33];
 
%% Performance specification transformation

%% calculate Yp1 such that there exists a permutation matrix PY giving
% PY*Yp=[Yp1;Yp2]
% with [Up;Yp1;Uf] full row rank;
% and rank([Up;Yp1;Uf])=rank([Up;Yp;Uf])

Yp1 = [];
Yp2 = Yp;

while 1 
   
    for i = 1:size(Yp2,1)
        if rank([Up;Yp1;Yp2(i,:);Uf])>rank([Up;Yp1;Uf])
            Yp1 = [Yp1;Yp2(i,:)];
            Yp2 = [Yp2(1:i-1,:);Yp2(i+1:end,:)];
            break
        end 
    end
    
    if rank([Up;Yp1;Uf])==rank([Up;Yp;Uf])
        break
    end 
end

%% calculate Bini, Bu, Bw, and y0

Lambda = [Up;Yp1;Uf];

Bini = Yf*(eye(size(Yf,2))+Lambda'*inv(Lambda*Lambda')*[zeros(size(Lambda,1)-Te*m,size(Uf,2));-Uf]);
Bu = Yf*Lambda'*inv(Lambda*Lambda')*[zeros(size(Lambda,1)-Te*m,Te*m);eye(Te*m)];

barBrho=[Bini*Md Bu Bini*M];

%% solve the optimization problem
% min_{alpha,u,gamma} gamma
% s.t., finalLMI>=0

ops = sdpsettings('solver','mosek'); 
ops = sdpsettings(ops,'verbose',0);
alphaw=sdpvar(1);
alphad=sdpvar(1);
alphayd=sdpvar(1);
alphayw=sdpvar(1);
alphaud=sdpvar(1);
alphauw=sdpvar(1);
optu=sdpvar(Te*m,1);
gamma=sdpvar(1,1);
epsilon_u = .5;  % \|u_i\|_2^2 \leq N*epsilon_u*m
epsilon_y = .5;  % \sum_i \|y_i\|_2^2 \leq N*epsilon_y*p

% generate matrix Qbar_g^a
r=zeros(Te*p,1);
% r=ones(N*p,1);
% r=[zeros(floor(N/2)*p,1);ones((N-floor(N/2))*p,1)];

barR=eye(Te*m);
barQ=eye(Te*p);
% Q = zeros(p);
% Q(1,1) = 1;
% barQ = kron(eye(N),Q);

Xi = [zeros(Te*m,size(Md,2)), eye(Te*m), zeros(Te*m,size(M,2))];


Qbarga_11 = gamma + r'*barQ*Bu*optu + optu'*Bu'*barQ*r - r'*barQ*r;
Qbarga_12 = -optu'*barR*Xi-(Bu*optu-r)'*barQ*barBrho;
Qbarga_22 = -Xi'*barR*Xi-barBrho'*barQ*barBrho;


Qbarga = [Qbarga_11  Qbarga_12;
          Qbarga_12' Qbarga_22];
     
% a_w_scale = 1e2;
% a_d_scale = 1e2;

% generate final LMI matrix
% tempSize=size(barR+Bu'*barQ*Bu,1);
zeroBlock=zeros(Te*m,size(Qbarga,2)-1);
finalLMI=[inv(barR+Bu'*barQ*Bu),  optu, zeroBlock;
          [optu';zeroBlock'],Qbarga-alphaw*barPhiw-alphad*barPhid];
      
% QUADRATIC CONSTRAINT ON THE WHOLE INPUT TRAJECTORY:  
% MEANS:
% LMIinputconst = [N*epsilon_u     optu' ;
%                     optu       eye(N*m)];

Psi_11 = Te*epsilon_u;
Psi_12 = zeros(1,Te*m);
Psi_22 = -eye(Te*m);

Psi = [Psi_11  Psi_12;
       Psi_12' Psi_22];

Psibara_11 = Psi_11 + Psi_12*optu + optu'*Psi_12';
Psibara_12 = (Psi_12 + optu'*Psi_22)*Xi;
Psibara_22 = Xi'*Psi_22*Xi;

Psibara = [Psibara_11  Psibara_12;
           Psibara_12' Psibara_22];

LMIinputconst = [-pinv(Psi_22) optu zeros(Te*m,size(barPhiw,2)-1);
                 [optu zeros(Te*m,size(barPhiw,2)-1)]' Psibara-alphaud*barPhid-alphauw*barPhiw];

% % BOX CONSTRAINTS ON EACH CONTROL INPUT:   TO MODIFY LATER IF NEEDED!!!
% 
% inputconst = cell(N,1);
% 
% for i=1:N
%     inputconst{i} = [epsilon_u  optu(i) ;
%                       optu(i)   eye(m)    ];
% end
% LMIinputconst = blkdiag(inputconst{:});

% OUTPUT CONSTRAINTS

Theta_11 = epsilon_y*Te;
Theta_12 = zeros(1,Te*p);
Theta_13 = -eye(Te*p);
Theta = [Theta_11  Theta_12;
         Theta_12' Theta_13];
     
Thetabara_11 = Theta_11 + Theta_12*Bu*optu + optu'*Bu'*Theta_12';
Thetabara_12 = (Theta_12 + optu'*Bu'*Theta_13)*barBrho;
Thetabara_22 = barBrho'*Theta_13*barBrho;

Thetabara = [Thetabara_11  Thetabara_12;
             Thetabara_12' Thetabara_22];
   
zeroBlockout=zeros(Te*m,size(barPhiw,2)-1);
LMIoutputconst = [-pinv(Bu'*Theta_13*Bu) optu zeroBlockout;
                  [optu'; zeroBlockout'] Thetabara-alphayd*barPhid-alphayw*barPhiw];

CONS1=[alphaw>=0];
CONS2=[alphad>=0];
CONS3=[finalLMI>=0];
CONS4=[LMIinputconst>=0];
CONS5=[alphauw>=0];
CONS6=[alphaud>=0];
CONS7=[LMIoutputconst>=0];
CONS8=[alphayw>=0];
CONS9=[alphayd>=0];
CONS=[CONS1,CONS2,CONS3,CONS4,CONS5,CONS6,CONS7,CONS8,CONS9];   % With I/O constraints
% CONS=[CONS1,CONS2,CONS3,CONS4,CONS5,CONS6];             % Only with I constraints
% CONS=[CONS1,CONS2,CONS3];             % Without I/O constraints

infosold = optimize(CONS,gamma*gamma,ops);
desiredControlInput = double(optu);
% desiredControlInput
% desiredControlInput.*desiredControlInput
desiredGamma=double(gamma);
alphaw = double(alphaw)
alphad = double(alphad)

if infosold.problem ~=0
    disp('Controller cannot be computed!')
%    return 
    continue
end

%% verification
% check that: using the above obtained desiredControlInput
% the performance guarantees for noiseless system can be obtained

optu=desiredControlInput;

% OCt = OC(p+1:(N+1)*p,:);
% HCt = HC(p+1:(N+1)*p,m+1:(N+1)*m);
% 
% xinit = x((Tini-1)*n+1:Tini*n);
% 
% opty=OCt*xinit+HCt*optu;
% 
% % x=zeros(3,N+1);
% % x(:,1) = xini;
% % opty=zeros(N,1);
% % for t=1:N
% %     x(:,t+1)=A*x(:,t)+B*optu(t);
% %     opty(t)=C*x(:,t+1);
% % end

OCt = OC(1:(Tini+Te)*p,:);
HCt = HC(1:(Tini+Te)*p,1:(Tini+Te)*m);

% x = Ot*xini + Ht*[uini;optu];
opty = OCt*xini + HCt*[uini;optu];
opty = opty(Tini*p+1:end);

figure
for j = 1:p
   subplot(p,1,j)
   plot(opty(j:p:end));
end


% disp(['u.*u = [', num2str(optu'.*optu'), ']^T <= ',num2str(epsilon_u),'?'])
disp(['u^T*u = ', num2str(optu'*optu), ' <= ',num2str(Te*epsilon_u),'?'])
disp(['y^T*y = ', num2str(opty'*opty), ' <= ',num2str(Te*epsilon_y),'?'])

noiselessGamma=optu'*barR*optu+(opty-r)'*barQ*(opty-r);
      
if ~((optu'*optu<=8) && (opty'*opty<=7))
    break
%     return
else 
    close all
end

end
%% randomly select a feasibility noise and check whether 
% the LQ performance is smaller than the previous obtained one

% generate feasible g0 and further obtain noise w
optgbar=sdpvar(size(barPhiw,1)-1,1);

CONS1=[[1;optgbar]'*barPhiw*[1;optgbar]>=1e-2];
CONS2=[[1;optgbar]'*barPhid*[1;optgbar]>=1e-2];
% CONS3=[optgbar(size(Mstar,2)+1:end-size(M,2))'*optgbar(size(Mstar,2)+1:end-size(M,2))>=1e-12];
CONS = [CONS1,CONS2];

infosolt = optimize(CONS,optgbar'*optgbar,ops);
if infosolt.problem ~= 0
    disp('No feasible gbar found!')
    return
end
desiredgbar = double(optgbar);

N_noisesamples = 100;
y_s = zeros(Te*p,N_noisesamples);
u_s = zeros(Te*m,N_noisesamples);
dbar_s = zeros(size(barA1d,1),N_noisesamples);
w_s = zeros(Tini*p,N_noisesamples);
gammas = zeros(N_noisesamples,1);

for indx = 1:N_noisesamples
   
    while 1
        desiredgbarn = desiredgbar + 0.02*(2*rand(size(desiredgbar))-1);
        consw = [1;desiredgbarn]'*barPhiw*[1;desiredgbarn];
        consd = [1;desiredgbarn]'*barPhid*[1;desiredgbarn];
        if (consw>=0) && (consd>=0)
            break;
        end
    end
    
    candidate_w=barA1w*desiredgbarn+barA2w;
    candidate_dbar=barA1d*desiredgbarn+barA2d;
    candidate_dini = candidate_dbar(1:end-Te*m);
    candidate_d = candidate_dbar(end-Te*m+1:end);
    
    dbar_s(:,indx) = candidate_dbar;
    w_s(:,indx) = candidate_w;
    
    ops = sdpsettings('solver','mosek'); 
    ops = sdpsettings(ops,'verbose',0);
    % optu=sdpvar(N*m,1);
    opty=sdpvar(Te*p,1);
    optg=sdpvar(size(Up,2),1);

    % generate matrix B8
    r=zeros(Te*p,1);
    barR=eye(Te*m);
    barQ=eye(Te*p);

    % generate final LMI matrix
    obj=(optu+candidate_d)'*barR*(optu+candidate_d)+(opty-r)'*barQ*(opty-r);

    CONSF=[[Up;Yp;Uf;Yf]*optg==[uini+candidate_dini;yini-candidate_w;optu+candidate_d;opty]];

    infosol = optimize(CONSF,[],ops);
    double(obj)
    disp(['y^T*y = ', num2str(double(opty)'*double(opty)), ' <= ',num2str(Te*epsilon_y),'?'])
    disp(['u^T*u = ', num2str((optu+candidate_d)'*(optu+candidate_d)), ' <= ',num2str(Te*epsilon_u),'?'])
    gammas(indx) = double(obj);
    y_s(:,indx) = double(opty);
    u_s(:,indx) = optu+candidate_d;
end


% Plot the output trajectories
int = 'Interpreter';
lat = 'Latex';
fonts = 'Fontsize';
fs = 14;
font = 'FontName';
fn = 'Times';
line = 'LineWidth';
Line = 2.5;

figure(2)
set(gca, font, fn, fonts, fs);
hold on
for j = 1:p
   subplot(p,1,j)
   plot(y_s(j:p:end,:));
   grid on
   ylabel(['$$\mathbf{y}_{',num2str(j),'}(k)$$'], font, fn, int,lat,fonts,fs)
   xlim([1 Te])
end

xlabel('$$k$$', font, fn, int,lat,fonts,fs)

% figure(3)
% set(gca, font, fn, fonts, fs);
% hold on
% for j = 1:m
%    subplot(m,1,j)
%    plot(u_s(j:m:end,:));
%    grid on
%    ylabel(['$$\mathbf{u}_{',num2str(j),'}(k)$$'], font, fn, int,lat,fonts,fs)
%    xlim([1 N])
% end
% 
% xlabel('$$k$$', font, fn, int,lat,fonts,fs)

inconsts = Te*epsilon_u-diag(u_s'*u_s);
outconsts = Te*epsilon_y-diag(y_s'*y_s);

figure(4)
set(gca, font, fn, fonts, fs);
subplot(3,1,1)
hold on
scatter(1:N_noisesamples,gammas,'b',line,Line);
plot([1 N_noisesamples],[desiredGamma desiredGamma],'r',line,Line);
grid on
ylabel(['$$\mathbf{\gamma}$$'], font, fn, int,lat,fonts,fs)
xlim([1 N_noisesamples])
subplot(3,1,2)
hold on
scatter(1:N_noisesamples,inconsts,'b',line,Line);
% plot([1 N_noisesamples],[N*epsilon_u N*epsilon_u],'r',line,Line);
grid on
ylabel(['$$\mathbf{\psi(\bar{u})}$$'], font, fn, int,lat,fonts,fs)
xlim([1 N_noisesamples])
subplot(3,1,3)
hold on
scatter(1:N_noisesamples,outconsts,'b',line,Line);
% plot([1 N_noisesamples],[N*epsilon_y N*epsilon_y],'r',line,Line);
grid on
ylabel(['$$\mathbf{\theta(y)}$$'], font, fn, int,lat,fonts,fs)
xlim([1 N_noisesamples])

xlabel('Number of noise and disturbance sample', font, fn, int,lat,fonts,fs)

all(gammas<desiredGamma)


% % desiredgbar = .1*(2*rand(size(barPhiw,1)-1,1)-1);
% % consw = [1;desiredgbar]'*barPhiw*[1;desiredgbar];
% % consd = [1;desiredgbar]'*barPhid*[1;desiredgbar];
% % 
% % while 1
% %     if ((consw>=0) && (consd>=0))
% %         break;
% %     end
% %     desiredgbar = rand(size(barPhiw,1)-1,1);
% %     consw = [1;desiredgbar]'*barPhiw*[1;desiredgbar];
% %     consd = [1;desiredgbar]'*barPhid*[1;desiredgbar];
% % end
% % 
% % consw
% % consd
% 
% candidate_w=barA1w*desiredgbar+barA2w;
% candidate_dbar=barA1d*desiredgbar+barA2d;
% candidate_dini = candidate_dbar(1:end-N*m);
% candidate_d = candidate_dbar(end-N*m+1:end);
% % candidate_w=[d;rho]
% % candidate_d = candidate_w(1:Tini*m)
% % candidate_rho = candidate_w(Tini*m+1:end)
% 
% % calculate the LQ performance corresponding to candidate_w
% 
% ops = sdpsettings('solver','mosek'); 
% ops = sdpsettings(ops,'verbose',0);
% % optu=sdpvar(N*m,1);
% opty=sdpvar(N*p,1);
% optg=sdpvar(size(Up,2),1);
% 
% % generate matrix B8
% r=zeros(N*p,1);
% barR=eye(N*m);
% barQ=eye(N*p);
% 
% % generate final LMI matrix
% obj=(optu+candidate_d)'*barR*(optu+candidate_d)+(opty-r)'*barQ*(opty-r);
% 
% CONSF=[[Up;Yp;Uf;Yf]*optg==[uini+candidate_dini;yini-candidate_w;optu+candidate_d;opty]];
% 
% infosol = optimize(CONSF,[],ops);
% trueGamma = double(obj);


desiredGamma
% trueGamma
noiselessGamma
infosold
% infosolt
max(abs(eig(A)))
noiselessGamma<desiredGamma
% trueGamma<desiredGamma
