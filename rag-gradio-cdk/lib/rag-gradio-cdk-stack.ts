import * as cdk from 'aws-cdk-lib';
import { Construct } from 'constructs';
import { ApplicationLoadBalancer } from "aws-cdk-lib/aws-elasticloadbalancingv2";
import { AutoScalingGroup, HealthCheck } from "aws-cdk-lib/aws-autoscaling";
import { Duration } from "aws-cdk-lib";
import {
  AmazonLinuxGeneration,
  AmazonLinuxImage,
  InstanceClass,
  InstanceSize,
  InstanceType,
  SubnetType,
  Vpc,
  IVpc,
} from "aws-cdk-lib/aws-ec2";

export class RagGradioCdkStack extends cdk.Stack {
  public readonly vpc: IVpc;
  public readonly autoScalingGroup: AutoScalingGroup;
  constructor(scope: Construct, id: string, props?: cdk.StackProps) {
    super(scope, id, props);
    this.vpc = new Vpc(this, "open-rag-vpc", {
      cidr: "10.0.0.1/24",
      subnetConfiguration: [
        {
          cidrMask: 28,
          name: "public subnet",
          subnetType: SubnetType.PUBLIC,
        },
        {
          cidrMask: 28,
          name: "private subnet",
          subnetType: SubnetType.PRIVATE_ISOLATED,
        },
      ],
    });

    const applicationAutoScalingGroup = new AutoScalingGroup(this, "AutoScalingGroup", {
      vpc: this.vpc,
      instanceType: InstanceType.of(InstanceClass.T3, InstanceSize.MICRO),
    });

    const applicationAutoScalingGroup2 = new AutoScalingGroup(this, "AutoScalingGroup", {
      vpc: this.vpc,
      instanceType: InstanceType.of(
        InstanceClass.BURSTABLE4_GRAVITON,
        InstanceSize.MICRO
      ),
      machineImage: new AmazonLinuxImage({
        generation: AmazonLinuxGeneration.AMAZON_LINUX_2,
      }),
      allowAllOutbound: true,
      maxCapacity: 2,
      minCapacity: 1,
      desiredCapacity: 1,
      spotPrice: "0.007", // $0.0032 per Hour when writing, $0.0084 per Hour on-demand
      healthCheck: HealthCheck.ec2(),
    });

    applicationAutoScalingGroup.scaleOnCpuUtilization("CpuScaling", {
        targetUtilizationPercent: 50,
        cooldown: Duration.minutes(1),
        estimatedInstanceWarmup: Duration.minutes(1),
    });

    this.autoScalingGroup = applicationAutoScalingGroup;

    const loadBalancer = new ApplicationLoadBalancer(this, "appLoadBalancer", {
      vpc: this.vpc,
      internetFacing: true,
    });

    const httpListener = loadBalancer.addListener("httpListener", {
      port: 80,
      open: true,
    });

    httpListener.addTargets('ApplicationSpotFleet', {
        port: 8080,
        targets: [this.autoScalingGroup],
    });
  }
}
