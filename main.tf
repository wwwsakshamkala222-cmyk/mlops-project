provider "aws" {
  region = "us-east-1"
}

# Create a small server (EC2) for our AI
resource "aws_instance" "mededge_server" {
  ami           = "ami-0c55b159cbfafe1f0" # Standard Linux
  instance_type = "t2.micro"             # Free tier eligible

  tags = {
    Name = "MedEdge-Production-Server"
  }
}

# Open the door so hospitals can send data
resource "aws_security_group" "allow_api" {
  name = "allow_api_traffic"
  ingress {
    from_port   = 8000
    to_port     = 8000
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
}