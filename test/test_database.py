import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent))

from utils import get_metric_db


def main():
    print(f"-- Test MetricDB (SQLite) --")
    print("=" * 40)
    print()

    try:
        db = get_metric_db()
        print("Get MetricDB singleton instance successfully")

        # Insert test metric
        db.insert_metric(step_type="test", current_epoch=1, total_epoch=8, loss=0.234, bleu=0.78)
        db.insert_metric(step_type="test", current_epoch=2, total_epoch=8, loss=0.210, bleu=0.79)
        db.insert_metric(step_type="test", current_epoch=3, total_epoch=8, loss=0.198, bleu=0.80)
        db.insert_metric(step_type="test", current_epoch=1, total_epoch=8, loss=0.186, bleu=0.81)
        db.insert_metric(step_type="test", current_epoch=2, total_epoch=8, loss=0.174, bleu=0.82)
        db.insert_metric(step_type="test", current_epoch=3, total_epoch=8, loss=0.162, bleu=0.83)
        print(f"\nInsert Operations:")
        print("Insert 6 test metric data successfully")

        # Query all metrics
        test_metrics  = db.query_metrics(step_type="test")
        # train_metrics = db.query_metrics(step_type="train")
        # eval_metrics  = db.query_metrics(step_type="eval")
        print(f"\nQuery Results:")
        print(f"Test  Metrics: {test_metrics}")
        # print(f"Train Metrics: {train_metrics}")
        # print(f"Eval  Metrics: {eval_metrics}")

        # Delete metrics
        delete_test  = db.delete_metric("test")
        print(f"\nDelete Operations:")
        print(f"Delete test metric data {'Successfully' if delete_test else 'Failed'}")

        # Query all metrics
        test_metrics  = db.query_metrics(step_type="test")
        # train_metrics = db.query_metrics(step_type="train")
        # eval_metrics  = db.query_metrics(step_type="eval")
        print(f"\nQuery Results:")
        print(f"Test  Metrics: {test_metrics}")
        # print(f"Train Metrics: {train_metrics}")
        # print(f"Eval  Metrics: {eval_metrics}")

    except Exception as e:
        print(f"MetricDB test failed: {str(e)}")
        return

    print("\n🎉 MetricDB test passed completely!")

if __name__ == "__main__":
    main()
