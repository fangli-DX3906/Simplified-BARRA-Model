%% loading the data
%%
firmdata=importdata('panel.csv');
firmdata=firmdata.data;
factors=importdata('riskfactor.csv');
factors=factors.data;

%% two strategies construction
%%
date=sort(unique(firmdata(:,1)));
initial=120;
predict_period=date((initial+1):end);
exret=nan(size(predict_period,1),2);
ticker=cell(size(predict_period,1),1);

for s=1:size(exret,1)

    basedate=date(s:(s+initial-1));
    
%%% step #1: fama-macbeth regression predicting the coefficient
    coeff=nan(6,size(basedate,1));
    residual=cell(initial,1);
    for i=1:size(basedate,1)
        temp=firmdata(firmdata(:,1)==basedate(i),:);
        residual{i}=nan(size(temp,1),2);
        residual{i}(:,1)=temp(:,2);
        [coeff(:,i),~,residual{i}(:,2)]=regress(temp(:,3), temp(:,4:9));
        clear temp
    end

%%% step #2-3: calcaulating the predicted coefficient
    m=nan(6,1);
    m=mean(coeff,2);
    v=nan(6);
    v=cov(coeff');

%%% step #4: calculating the predicted idiosyncratic volatility
    current=firmdata(firmdata(:,1)==basedate(end),:);
    csign_whole=current(:,2);
    cpt_idvol=nan(initial+1,size(csign_whole,1));
    cpt_idvol(1,:)=csign_whole';

    for i=1:size(cpt_idvol,2)
        for t=1:120
            if isempty( find( residual{t}(:,1)==cpt_idvol(1,i) ) )
                cpt_idvol(t+1,i)=nan;
            else
                cpt_idvol(t+1,i)=residual{t}( find(residual{t}(:,1)==cpt_idvol(1,i)) , 2 );
            end
        end
    end

    csign=nan(size(csign_whole,1),3);
    csign(:,1)=csign_whole';

    % eliminate the stock with less than 80 residual obervations 
    for i=1:size(csign,1)
        c=0;
        for t=1:initial
            if ~isnan(cpt_idvol(t+1,i))
                c=c+1;
            end
        end
        csign(i,2)=c;
    end
    csign=csign(csign(:,2)>=80,:);

    % eliminate stock with missing characters
    for i=1:size(csign,1)
        temp=current(current(:,2)==csign(i,1),:);
        c=true;
        for j=1:6
            c=c & ( ~isnan( temp( 3+j ) ) );
        end
        if c
            csign(i,3)=csign(i,1);
        else
            csign(i,3)=nan;
        end
        clear temp c
    end
    csign=csign(~isnan(csign(:,3)),:);

    % eliminate stock with missing next period return
    next=firmdata(firmdata(:,1)==date(s+initial),:);
    next=next(ismember(next(:,2),csign(:,1)),:);
    next=next(~isnan(next(:,3)),:);
    nsign=next(:,2);
    bothsign=intersect(csign(:,1),nsign);
    csign=csign(ismember(csign(:,1),bothsign),:);
    ticker{s}=csign(:,1);   % ticker records the remaining stocks at end of period

    % calculate the idiosyncratic volatility
    idvol=zeros(size(ticker{s},1),size(ticker{s},1));
    for i=1:size(ticker{s},1)
        temp=cpt_idvol(2:end, cpt_idvol(1,:)==ticker{s}(i));
        temp=temp(~isnan(temp));
        idvol(i,i)=(std(temp,1))^2;
        clear temp
    end

%%% step #5-6: calculating the individual forecast mean and variance
    cpt_stat=current(ismember(current(:,2),ticker{s}),:);
    projm=cpt_stat(:,4:end)*m;
    projv=cpt_stat(:,4:end)*v*cpt_stat(:,4:end)'+idvol;

%%% step #7a: mean-variance optimization
    e=ones(size(projm,1),1);
    w_tan=(projv \ projm)/(e'*(projv \ projm));
    next=next(ismember(next(:,2),ticker{s}),:);
    exret(s,1)=w_tan'*next(:,3);

%%% step #7b: optimized the objective function
    lambda=3;
    Aeq=[ones(1,size(cpt_stat,1));
              cpt_stat(:,4)';
              cpt_stat(:,5)';];
    beq=[0;0;0];  
    lb=-0.03*ones(size(cpt_stat,1),1);
    ub=0.03*ones(size(cpt_stat,1),1);
    f=(-1)*projm/lambda;
    H=projv;

    [w,fval,exisflag,output] = quadprog(H,f,[],[],Aeq,beq,lb,ub);
    exret(s,2)=w'*next(:,3);

    clear m v
    
    s

end

%% performance evaluation
%%
mkt_ret=factors((initial+1):end ,2);
portfolio=nan(size(predict_period,1),4);
portfolio(:,1)=factors((initial+1):end ,1);
portfolio(:,2)=exret(:,1);
portfolio(:,3)=exret(:,2)+mkt_ret;
portfolio(:,4)=mkt_ret;

% (1) annualized average excess return
aexret=mean(portfolio(:,2:4))*12;

% (2) annualized std of excess return
astd=std(portfolio(:,2:4))*sqrt(12);

% (3) annualized sharpe ratio
asp=aexret./astd;

% (4) annualized capm alpha and its t-statistics
eps=zeros(size(portfolio,1),2);
capm_alpha=nan(2,1);
t_stat=nan(2,1);
abeta=nan(2,1);
rsq=nan(2,1);
for i=1:2   
    [b,~,eps(:,i)]=regress(portfolio(:,i+1), [ones(size(portfolio,1),1) portfolio(:,4)]);
    capm_alpha(i)=b(1)*12;
    stats=regstats(portfolio(:,i+1), portfolio(:,4) ,'linear');
    t_stat(i)=stats.tstat.t(1);
    abeta(i)=b(2);
    rsq(i)=stats.rsquare;
end

% (5) beta and annualized systematic volatility
asysvol=abeta*std(portfolio(:,4))*sqrt(12);

% (6) annualized TEV
atev=std(eps)*sqrt(12);

% (7) R-square of regression 
rsq;

% (8) information ratio
ir=(capm_alpha'/12) ./std(eps);

% (9) maximal DD
maxdd=zeros(1,2);
indd=nan(1,2);
cumret=zeros(size(portfolio,1),2);
for n=2:3
    for i=1:size(cumret,1)
        if i==1
            cumret(i,n-1)=portfolio(i,n);
        else 
            product=1;
            for j=1:i
                product=product*(portfolio(j,n)+1);
            end
            cumret(i,n-1)=product-1;
        end
    end
end

plot(cumret(:,1),'b')
hold on
plot(cumret(:,2),'r')

record=nan(size(cumret,1),2,2);
for n=2:3
    for i=1:size(cumret,1)
        [M,I]=min(cumret(i:end,n-1));
        record(i,2,n-1)=I;
        abI=I+(i-1);
        record(i,1,n-1)=cumret(i,n-1)-M;
    end
    [maxdd(n-1),indd(n-1)]=max(record(:,1,n-1));
end

% (10) maximal RP
maxrp=zeros(1,2);
indrp=zeros(1,2);
rp=nan(size(cumret,1),2);
for n=2:3
    for i=1:size(cumret,1)
        order=find(cumret((i+1):end,n-1)>=cumret(i,n-1));
        if isempty(order)
            rp(i,n-1)=size(cumret,1)-i;
        else
            rp(i,n-1)=order(1);
        end
    end
    [maxrp(n-1),indrp(n-1)]=max(rp(:,n-1));
end