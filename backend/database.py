from sqlalchemy import create_engine, text

DATABASE_URL = "mysql+pymysql://root:@localhost/learningloop"

engine = create_engine(
    DATABASE_URL
)

try:
    connection = engine.connect()
    print("Database connected successfully!")
    connection.close()

except Exception as e:
    print("Database connection failed!")
    print(e)

    from sqlalchemy import text


def save_prediction(
    smiles,
    prediction,
    confidence
):

    with engine.connect() as connection:

        query = text("""
            INSERT INTO predictions
            (
                smiles,
                prediction,
                confidence
            )
            VALUES
            (
                :smiles,
                :prediction,
                :confidence
            )
        """)

        connection.execute(
            query,
            {
                "smiles": smiles,
                "prediction": prediction,
                "confidence": confidence
            }
        )

        connection.commit()

        print("Prediction saved!")
        

from math import ceil

def get_predictions(page=1, limit=10):

    offset = (page - 1) * limit

    with engine.connect() as connection:

        count_query = text("""
            SELECT COUNT(*) as total
            FROM predictions
        """)

        total = connection.execute(
                count_query
            ).scalar()

        query = text("""
            SELECT *
            FROM predictions
            ORDER BY id DESC
            LIMIT :limit
            OFFSET :offset
        """)

        result = connection.execute(
            query,
            {
                "limit": limit,
                "offset": offset
            }
        )

        rows = []

        for row in result:

            rows.append({
                "id": row.id,
                "smiles": row.smiles,
                "prediction": row.prediction,
                "confidence": float(row.confidence),
                "created_at": str(row.created_at)
            })

        return {
            "data": rows,
            "total": total,
            "page": page,
            "pages": ceil(total / limit)
        }
    

def get_dashboard_stats():

    with engine.connect() as connection:

        total_query = text("""
            SELECT COUNT(*) as total
            FROM predictions
        """)

        active_query = text("""
            SELECT COUNT(*) as total
            FROM predictions
            WHERE prediction = 'Active'
        """)

        inactive_query = text("""
            SELECT COUNT(*) as total
            FROM predictions
            WHERE prediction = 'Inactive'
        """)

        total = connection.execute(
                total_query
            ).scalar()

        active = connection.execute(
                active_query
            ).scalar()

        inactive = connection.execute(
                inactive_query
            ).scalar()

        return {
            "total_predictions": total,
            "active_predictions": active,
            "inactive_predictions": inactive
        }
    

def get_top_compounds():

    with engine.connect() as connection:

        query = text("""
            SELECT
                smiles,
                COUNT(*) as total
            FROM predictions
            GROUP BY smiles
            ORDER BY total DESC
            LIMIT 10
        """)

        result = connection.execute(query)

        rows = []

        for row in result:

            rows.append({
                "smiles": row.smiles,
                "count": row.total
            })

        return rows