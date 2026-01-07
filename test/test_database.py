import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent))
from utils import get_metric_db


def main():
    print(f"-- Test Database Module --")
    print("=" * 40)
    print()

    try:
        db = get_metric_db()
        print("Get MetricDB singleton instance successfully")

        # Insert test metric (parameters in one line)
        db.insert_metric(step_type="test", stage_epoch=1, total_stage_epoch=8, task_epoch=1, total_task_epoch=1, loss=0.234, bleu=0.78)
        db.insert_metric(step_type="test", stage_epoch=2, total_stage_epoch=8, task_epoch=1, total_task_epoch=1, loss=0.210, bleu=0.79)
        db.insert_metric(step_type="test", stage_epoch=3, total_stage_epoch=8, task_epoch=1, total_task_epoch=1, loss=0.198, bleu=0.88)
        db.insert_metric(step_type="test", stage_epoch=2, total_stage_epoch=8, task_epoch=1, total_task_epoch=1, loss=0.186, bleu=0.81)
        db.insert_metric(step_type="test", stage_epoch=4, total_stage_epoch=8, task_epoch=1, total_task_epoch=1, loss=0.174, bleu=0.82)
        db.insert_metric(step_type="test", stage_epoch=6, total_stage_epoch=8, task_epoch=1, total_task_epoch=1, loss=0.162, bleu=0.83)
        db.insert_metric(step_type="test", stage_epoch=0, total_stage_epoch=0, task_epoch=1, total_task_epoch=4, loss=0.234, bleu=0.78)
        db.insert_metric(step_type="test", stage_epoch=0, total_stage_epoch=0, task_epoch=2, total_task_epoch=4, loss=0.211, bleu=0.79)
        db.insert_metric(step_type="test", stage_epoch=0, total_stage_epoch=0, task_epoch=3, total_task_epoch=4, loss=0.198, bleu=0.88)
        print(f"\nInsert Operations:")
        print("Insert 6 test metric data successfully")

        # Query all metrics
        test_metrics  = db.query_metrics(step_type="test")
        print(f"\nQuery Results:")
        [print(f"  Test Metric {i+1}: {metric}") for i, metric in enumerate(test_metrics)]

        # Delete metrics
        delete_test  = db.delete_metric("test")
        print(f"\nDelete Operations:")
        print(f"Delete test metric data {'Successfully' if delete_test else 'Failed'}")

        # Query all metrics
        test_metrics  = db.query_metrics(step_type="test")
        print(f"\nQuery Results:")
        [print(f"  Test Metric {i+1}: {metric}") for i, metric in enumerate(test_metrics)]

    except Exception as e:
        print(f"MetricDB test failed: {str(e)}")
        return

    print("\n🎉 Database Module test passed completely!")


if __name__ == "__main__":
    main()


