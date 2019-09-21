using System.Net.Sockets;
using UnityEngine;

public class Connect
{
    Socket sender;
    byte[] messageHolder;
    const int Length = 8;//95 * sizeof(float);

    public void Initial(int port)
    {
        Quit();
        messageHolder = new byte[Length];
        sender = new Socket(AddressFamily.InterNetwork, SocketType.Stream, ProtocolType.Tcp);
        sender.Connect("localhost", port);
        Debug.Log("send initial msg");
        sender.Send(new byte[] { 0x00, 0x11, 0x22 });
    }

    public void Receive()
    {
        if (sender.Receive(messageHolder) > 0)
        {
            Debug.Log("recv msg");
            string str = "";
            for (int i = 0; i < Length; i++)
            {
                str += messageHolder[i].ToString("x2") + " ";
            }
            Debug.Log(str);
        }
    }

    public void Quit()
    {
        if (sender != null)
        {
            Debug.Log("Socket close");
            sender.Close();
            sender = null;
        }
    }


}
