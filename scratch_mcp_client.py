import asyncio
import sys

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from rag.config import settings

#system arguments for testing
#Actual code should be
#TOKEN = settings.mcp_auth_token
TOKEN = sys.argv[1] if len(sys.argv) > 1 else settings.mcp_auth_token

def _show(err):
    subs = getattr(err, "exceptions", None)   # ExceptionGroup? recurse into it
    if subs:
        for s in subs:
            _show(s)
    else:
        print("BLOCKED:", type(err).__name__, "-", str(err)[:160])

async def main():
    url = "http://127.0.0.1:8001/mcp"
    headers = {"Authorization": f"Bearer {TOKEN}"}
    try:
        async with streamablehttp_client(url, headers=headers) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = await session.list_tools()
                print("tools:", [t.name for t in tools.tools])
                r = await session.call_tool("search_documents", {"query": "what is chunking?"})
                print("result:", r.content[0].text[:150])
                
                r = await session.call_tool("delete_document", {"document_id": 999})
                print("PREVIEW:", r.content[0].text)

                r = await session.call_tool("delete_document", {"document_id": 999, "confirm": True})
                print("CONFIRMED:", r.content[0].text)

    except Exception as e:
        _show(e)


asyncio.run(main())
