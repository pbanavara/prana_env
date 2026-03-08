import asyncio
from prana_env.client import PranaEnv
from prana_env.models import PranaAction

async def main():
    async with PranaEnv(base_url="http://localhost:8000") as client:
        await client.reset()
        result = await client.step(PranaAction(
            action_type="query_db",
            target="PatientDB",
            field="hba1c",
            patient_id="P001",
        ))
        print(result.observation.query_result)

asyncio.run(main())
